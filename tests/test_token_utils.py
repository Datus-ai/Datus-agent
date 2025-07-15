from datus.prompts.schema_lineage import gen_prompt
from datus.utils.constants import DBType
from datus.utils.token_utils import cal_gpt_tokens


def test_token_calculation():
    prompt = gen_prompt(top_n=10, dialect=DBType.SNOWFLAKE)

    gpt_tokens = cal_gpt_tokens(prompt[0]["content"] + prompt[1]["content"])
    print(f"gpt_tokens: {gpt_tokens}")
    # deepseek_tokens = cal_deepseek_tokens(prompt)
