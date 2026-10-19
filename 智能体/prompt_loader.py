from 智能体.config_handler import prompt_conf
from 智能体.path_toll import get_abs_path
from 智能体.logger_hander import logger


def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompt_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_rag_summarize_prompts():
    try:
        rag_summarize_prompt_path = get_abs_path(prompt_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_summarize_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_summarize_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_summarize_prompts]解析系统提示词出错，{str(e)}")
        raise e
def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompt_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_sentiment_prompts():
    try:
        sentiment_prompt_path = get_abs_path(prompt_conf["sentiment_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_sentiment_prompts]在yaml配置项中没有sentiment_prompt_path配置项")
        raise e

    try:
        return open(sentiment_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_sentiment_prompts]解析系统提示词出错，{str(e)}")
        raise e

def load_category_prompts():
    try:
        category_prompt_path = get_abs_path(prompt_conf["category_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_category_prompts]在yaml配置项中没有category_prompt_path配置项")
        raise e

    try:
        return open(category_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_category_prompts]解析系统提示词出错，{str(e)}")
        raise e

if __name__ == "__main__":
    print(load_rag_summarize_prompts())
    # print(load_report_prompts())
    # print(load_system_prompts())

