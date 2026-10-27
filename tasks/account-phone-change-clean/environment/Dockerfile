FROM python:3.12-slim
WORKDIR /app
# Fixture data the mock MCP tools serve. The persona is read by the
# user-simulator in the agent adapter, never exposed as a tool.
COPY fixtures.json persona.json /app/
RUN pip install --no-cache-dir pyyaml
