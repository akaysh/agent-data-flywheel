import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from llm.cache import CacheMissError, DiskLLMCache


def fake_model(cache, replies):
    return GenericFakeChatModel(messages=iter(replies), cache=cache)


def test_second_identical_call_is_served_from_disk(tmp_path):
    cache = DiskLLMCache(tmp_path, offline=False)
    first = fake_model(cache, [AIMessage(content="hello")]).invoke([HumanMessage(content="hi")])

    # a fresh model with no replies left can only answer from the cache
    replay = fake_model(DiskLLMCache(tmp_path, offline=False), []).invoke(
        [HumanMessage(content="hi")])
    assert first.content == replay.content == "hello"
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_offline_miss_raises_instead_of_calling_the_model(tmp_path):
    cache = DiskLLMCache(tmp_path, offline=True)
    with pytest.raises(CacheMissError):
        fake_model(cache, [AIMessage(content="x")]).invoke([HumanMessage(content="never seen")])


def test_offline_hit_replays_recorded_response(tmp_path):
    fake_model(DiskLLMCache(tmp_path, offline=False), [AIMessage(content="recorded")]).invoke(
        [HumanMessage(content="q")])
    replay = fake_model(DiskLLMCache(tmp_path, offline=True), []).invoke([HumanMessage(content="q")])
    assert replay.content == "recorded"


def test_tool_calls_survive_the_round_trip(tmp_path):
    tool_msg = AIMessage(content="", tool_calls=[
        {"name": "get_charges", "args": {"ride_id": "R1"}, "id": "call_1", "type": "tool_call"}])
    fake_model(DiskLLMCache(tmp_path, offline=False), [tool_msg]).invoke([HumanMessage(content="q")])
    replay = fake_model(DiskLLMCache(tmp_path, offline=True), []).invoke([HumanMessage(content="q")])
    assert replay.tool_calls[0]["name"] == "get_charges"
    assert replay.tool_calls[0]["args"] == {"ride_id": "R1"}


def test_different_prompts_get_different_entries(tmp_path):
    cache = DiskLLMCache(tmp_path, offline=False)
    model = fake_model(cache, [AIMessage(content="a"), AIMessage(content="b")])
    model.invoke([HumanMessage(content="one")])
    model.invoke([HumanMessage(content="two")])
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_hits_are_appended_to_the_hit_log_when_configured(tmp_path, monkeypatch):
    log = tmp_path / "hits.log"
    monkeypatch.setenv("CACHE_HIT_LOG", str(log))
    cache_dir = tmp_path / "c"
    fake_model(DiskLLMCache(cache_dir, offline=False), [AIMessage(content="a")]).invoke(
        [HumanMessage(content="q")])
    fake_model(DiskLLMCache(cache_dir, offline=True), []).invoke([HumanMessage(content="q")])
    entry = next(cache_dir.glob("*.json")).name
    assert entry in log.read_text().split()


def test_prune_removes_entries_not_in_the_hit_log(tmp_path):
    from llm.cache import prune_unused
    (tmp_path / "keep.json").write_text("{}")
    (tmp_path / "drop.json").write_text("{}")
    hits = tmp_path / "hits.log"
    hits.write_text("keep.json\nkeep.json\n")
    assert prune_unused(tmp_path, hits) == 1
    assert [p.name for p in tmp_path.glob("*.json")] == ["keep.json"]


def test_prune_keeps_underscore_metadata_files(tmp_path):
    from llm.cache import prune_unused
    (tmp_path / "_models.json").write_text("{}")
    (tmp_path / "old.json").write_text("{}")
    hits = tmp_path / "hits.log"
    hits.write_text("")
    prune_unused(tmp_path, hits)
    assert [p.name for p in tmp_path.glob("*.json")] == ["_models.json"]


def test_cache_key_ignores_the_endpoint_so_replay_works_without_credentials(tmp_path):
    from llm.cache import normalize_llm_string
    online = ('{"kwargs": {"model_name": "m", "openai_api_base": "https://internal.example/v1", '
              '"temperature": 0.0}}---[(\'stop\', None)]')
    offline = '{"kwargs": {"model_name": "m", "stream_usage": true, "temperature": 0.0}}---[(\'stop\', None)]'
    assert normalize_llm_string(online) == normalize_llm_string(offline)


def test_cache_key_still_distinguishes_models_and_temperature():
    from llm.cache import normalize_llm_string
    a = '{"kwargs": {"model_name": "a", "temperature": 0.0}}---[]'
    assert normalize_llm_string(a) != normalize_llm_string(a.replace('"a"', '"b"'))
    assert normalize_llm_string(a) != normalize_llm_string(a.replace("0.0", "0.7"))


def test_key_ignores_volatile_message_metadata(tmp_path):
    """A fresh response and the same response restored from cache must produce the same key."""
    fresh = AIMessage(content="hi", tool_calls=[
        {"name": "t", "args": {"a": 1}, "id": "c1", "type": "tool_call"}],
        response_metadata={"token_usage": {"total_tokens": 99}, "finish_reason": "tool_calls"},
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}, id="run-abc")
    restored = AIMessage(content="hi", tool_calls=[
        {"name": "t", "args": {"a": 1}, "id": "c1", "type": "tool_call"}])
    cache = DiskLLMCache(tmp_path, offline=False)
    from langchain_core.load import dumps
    assert cache._path(dumps([fresh]), "{}---") == cache._path(dumps([restored]), "{}---")


def test_key_still_distinguishes_different_conversations(tmp_path):
    from langchain_core.load import dumps
    cache = DiskLLMCache(tmp_path, offline=False)
    assert cache._path(dumps([HumanMessage(content="a")]), "{}---") != cache._path(
        dumps([HumanMessage(content="b")]), "{}---")
    ai_a = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1", "type": "tool_call"}])
    ai_b = AIMessage(content="", tool_calls=[{"name": "u", "args": {}, "id": "1", "type": "tool_call"}])
    assert cache._path(dumps([ai_a]), "{}---") != cache._path(dumps([ai_b]), "{}---")


def test_non_message_prompts_are_used_as_is(tmp_path):
    cache = DiskLLMCache(tmp_path, offline=False)
    assert cache._path("plain text prompt", "{}---") != cache._path("another", "{}---")


def test_gateway_subclass_keys_like_plain_chatopenai(monkeypatch, tmp_path):
    from langchain_openai import ChatOpenAI

    from llm.cache import normalize_llm_string
    from llm.factory import GatewayChatModel

    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    args = dict(model="m", api_key="k", temperature=0)
    assert (normalize_llm_string(GatewayChatModel(**args)._get_llm_string())
            == normalize_llm_string(ChatOpenAI(**args)._get_llm_string()))
