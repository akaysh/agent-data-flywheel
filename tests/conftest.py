from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class ScriptedChatModel(GenericFakeChatModel):
    """Replays scripted AIMessages and tolerates bind_tools()."""

    def bind_tools(self, tools, **kwargs):
        return self
