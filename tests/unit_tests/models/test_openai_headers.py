from datus.configuration.agent_config import load_model_config


def test_model_config_headers():
    config_data = {
        "type": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
        "model": "gpt-4",
        "default_headers": {"X-Custom-Header": "test-value"},
    }
    config = load_model_config(config_data)
    assert config.default_headers == {"X-Custom-Header": "test-value"}
