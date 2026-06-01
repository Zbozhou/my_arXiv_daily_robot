import os
import re
import json
import arxiv
import yaml
import logging
import argparse
import datetime
import requests
import subprocess
import time

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# ======= 常量 =======
# Hugging Face：用 arxiv_id 映射 Hub 上的 spaces/models/datasets
HF_REPOS_API = "https://huggingface.co/api/arxiv/{arxiv_id}/repos"
HF_HEADERS = {"User-Agent": "arxiv-daily/1.0"}

# GitHub 搜索（兜底）
GITHUB_SEARCH_REPO = "https://api.github.com/search/repositories"
GITHUB_SEARCH_CODE = "https://api.github.com/search/code"
GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "arxiv-daily/1.0"
}
if os.getenv("GITHUB_TOKEN"):
    GH_HEADERS["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN')}"

# arXiv 页面
arxiv_url = "https://arxiv.org/"

# ======= 工具函数 =======

def load_config(config_file:str) -> dict:
    """
    读取配置，并把 keywords 下的 filters/require 拼成 arXiv 查询串：
    filters 内部 OR，require 内部 OR，二者之间 AND。
    """
    def pretty_filters(**config) -> dict:
        keywords = {}
        OR = ' OR '
        FIELD = 'all:'

        def quote_if_needed(s: str) -> str:
            s = s.strip()
            return f"\"{s}\"" if (' ' in s or '-' in s) else s

        def parse_filters(filters: list) -> str:
            terms = []
            for flt in filters:
                terms.append(FIELD + quote_if_needed(flt))
            return OR.join(terms)

        def group_filters(filters: list) -> str:
            query = parse_filters(filters)
            if not query:
                return ''
            return f"({query})" if OR in query else query

        for k,v in config['keywords'].items():
            if v.get('query'):
                keywords[k] = v['query']
                continue

            filter_query = group_filters(v.get('filters', []))
            required_query = group_filters(v.get('require', []))
            if required_query and filter_query:
                keywords[k] = f"{required_query} AND {filter_query}"
            else:
                keywords[k] = filter_query or required_query
        return keywords

    with open(config_file,'r') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
        config['kv'] = pretty_filters(**config)
        logging.info(f'config = {config}')
    return config

def get_authors(authors, first_author = False):
    if not authors:
        return ""
    if first_author:
        return str(authors[0])
    return ", ".join(str(author) for author in authors)

def sort_papers(papers):
    output = {}
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output

def http_get(url, headers=None, params=None, timeout=10, retries=2, sleep=0.8):
    """ 简单 GET 带重试 """
    last_exc = None
    for _ in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            else:
                logging.warning(f"GET {url} status={r.status_code} params={params}")
        except Exception as e:
            last_exc = e
            logging.warning(f"GET {url} exception: {e}")
        time.sleep(sleep)
    if last_exc:
        raise last_exc
    return None

def get_code_link(qword:str) -> str | None:
    """
    用 GitHub 仓库搜索找一个可能的实现（按 stars 降序）。
    @param qword: 论文标题或 arxiv id
    @return 仓库 html_url 或 None
    """
    params = {
        "q": qword,
        "sort": "stars",
        "order": "desc",
        "per_page": 5
    }
    try:
        r = http_get(GITHUB_SEARCH_REPO, headers=GH_HEADERS, params=params, timeout=10)
        if not r:
            return None
        results = r.json()
        items = results.get("items", [])
        if items:
            return items[0].get("html_url")
    except Exception as e:
        logging.error(f"GitHub search error: {e}")
    return None

def find_code_repo(paper_title: str, arxiv_id_no_ver: str, primary_author: str | None = None) -> str | None:
    """
    更智能的 GitHub 兜底：
    1) 用标题短语搜 README/描述
    2) 再用 arXiv ID 搜
    3) 再用 Code Search 在 README 文件里搜 arXiv ID
    """
    try:
        # 1) 标题短语搜索
        q1 = f"\"{paper_title}\" in:readme,in:description"
        r = http_get(GITHUB_SEARCH_REPO, headers=GH_HEADERS,
                     params={"q": q1, "sort": "stars", "order": "desc", "per_page": 5}, timeout=10)
        if r and r.json().get("items"):
            return r.json()["items"][0]["html_url"]

        # 2) arXiv ID 搜索
        q2 = f"\"{arxiv_id_no_ver}\" in:name,readme,description"
        r = http_get(GITHUB_SEARCH_REPO, headers=GH_HEADERS,
                     params={"q": q2, "sort": "stars", "order": "desc", "per_page": 5}, timeout=10)
        if r and r.json().get("items"):
            return r.json()["items"][0]["html_url"]

        # 3) Code Search：README 中包含 arXiv ID
        q3 = f"\"{arxiv_id_no_ver}\" in:file filename:README"
        r = http_get(GITHUB_SEARCH_CODE, headers=GH_HEADERS,
                     params={"q": q3, "per_page": 5}, timeout=10)
        if r and r.json().get("items"):
            return r.json()["items"][0]["repository"]["html_url"]
    except Exception as e:
        logging.error(f"find_code_repo error: {e}")
    return None

def get_repo_from_hf(arxiv_id_no_ver: str) -> str | None:
    """
    从 Hugging Face Hub 获取与论文关联的 spaces/models/datasets。
    优先选择：Spaces -> Models -> Datasets
    返回对应的 Hub 链接，失败返回 None。
    """
    url = HF_REPOS_API.format(arxiv_id=arxiv_id_no_ver)
    try:
        r = http_get(url, headers=HF_HEADERS, timeout=10)
        if not r:
            return None
        data = r.json()  # {"models":[...], "datasets":[...], "spaces":[...]}

        def pick(arr, t):
            for it in (arr or []):
                rid = it.get("id")  # 形如 "org/name"
                if rid:
                    return f"https://huggingface.co/{t}/{rid}"
            return None

        return (pick(data.get("spaces"), "spaces")
                or pick(data.get("models"), "models")
                or pick(data.get("datasets"), "datasets"))
    except Exception as e:
        logging.error(f"HF repos error: {e}")
        return None

def _iter_arxiv_results(query: str, n: int):
    """
    封装 arXiv 查询，兼容 arxiv 包的新旧结果迭代 API。
    遇到 UnexpectedEmptyPageError 降级到 ≤25 条再拉；遇到 HTTP 429 时退避重试。
    """
    def parse_backoff_seconds() -> list[int]:
        raw = os.getenv("ARXIV_429_BACKOFF_SECONDS", "30,90,180")
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            return [30, 90, 180]

    def iter_results(search, page_size):
        if hasattr(search, "results"):
            return search.results()
        client_kwargs = {
            "page_size": max(1, min(page_size, n)),
            "delay_seconds": float(os.getenv("ARXIV_DELAY_SECONDS", "8")),
            "num_retries": int(os.getenv("ARXIV_NUM_RETRIES", "5")),
        }
        try:
            client = arxiv.Client(**client_kwargs)
        except TypeError:
            client = arxiv.Client()
        return client.results(search)

    def is_429_error(err) -> bool:
        status = getattr(err, "status", None) or getattr(err, "status_code", None)
        return status == 429 or "HTTP 429" in str(err)

    empty_page_error = getattr(arxiv, "UnexpectedEmptyPageError", None)
    http_error = getattr(arxiv, "HTTPError", None)

    def is_empty_page_error(err) -> bool:
        return empty_page_error is not None and isinstance(err, empty_page_error)

    def is_http_error(err) -> bool:
        return http_error is not None and isinstance(err, http_error)

    for max_results in [n, min(n, 25)]:
        backoff_seconds = [0] + parse_backoff_seconds()
        for attempt, delay in enumerate(backoff_seconds):
            if delay:
                logging.warning(f"arXiv HTTP 429; sleeping {delay}s before retry")
                time.sleep(delay)

            try:
                se = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                )
                for r in iter_results(se, max_results):
                    yield r
                return
            except Exception as err:
                if not is_empty_page_error(err):
                    if is_http_error(err) and is_429_error(err):
                        if attempt == len(backoff_seconds) - 1:
                            logging.error("arXiv HTTP 429 persisted; skipping this query for now")
                            return
                        continue
                    raise

                if max_results <= 25:
                    raise
                logging.warning("Empty page from arXiv; retrying with fewer results (<=25)")
                break

def get_daily_papers(topic,query="slam", max_results=2):
    """
    @param topic: str
    @param query: str
    @return paper_with_code: dict
    """
    content = {}
    content_to_web = {}

    for result in _iter_arxiv_results(query, max_results):

        paper_id            = result.get_short_id()         # 例如 2108.09112v1
        paper_title         = result.title
        paper_url           = result.entry_id
        paper_abstract      = (result.summary or "").replace("\n"," ")
        paper_authors       = get_authors(result.authors)
        paper_first_author  = get_authors(result.authors,first_author = True)
        primary_category    = result.primary_category
        publish_time        = result.published.date() if result.published else ""
        update_time         = result.updated.date() if result.updated else publish_time
        comments            = result.comment

        logging.info(f"Time = {update_time} title = {paper_title} author = {paper_first_author}")

        # 去掉版本号：2108.09112v1 -> 2108.09112
        ver_pos = paper_id.find('v')
        paper_key = paper_id if ver_pos == -1 else paper_id[:ver_pos]
        paper_url = arxiv_url + 'abs/' + paper_key

        # 先尝试 HF，失败再 GitHub 搜索作为兜底
        repo_url = get_repo_from_hf(paper_key)
        if repo_url is None:
            repo_url = (find_code_repo(paper_title, paper_key, paper_first_author)
                        or get_code_link(paper_title)
                        or get_code_link(paper_key))

        try:
            if repo_url is not None:
                content[paper_key] = "|**{}**|**{}**|{} et.al.|[{}]({})|**[link]({})**|\n".format(
                       update_time,paper_title,paper_first_author,paper_key,paper_url,repo_url)
                content_to_web[paper_key] = "- {}, **{}**, {} et.al., Paper: [{}]({}), Code: **[{}]({})**".format(
                       update_time,paper_title,paper_first_author,paper_url,paper_url,repo_url,repo_url)
            else:
                content[paper_key] = "|**{}**|**{}**|{} et.al.|[{}]({})|null|\n".format(
                       update_time,paper_title,paper_first_author,paper_key,paper_url)
                content_to_web[paper_key] = "- {}, **{}**, {} et.al., Paper: [{}]({})".format(
                       update_time,paper_title,paper_first_author,paper_url,paper_url)

            comments = None  # TODO: 保留注释逻辑
            if comments != None:
                content_to_web[paper_key] += f", {comments}\n"
            else:
                content_to_web[paper_key] += f"\n"

        except Exception as e:
            logging.error(f"exception: {e} with id: {paper_key}")

    data = {topic:content}
    data_web = {topic:content_to_web}
    return data,data_web

def update_paper_links(filename):
    '''
    weekly update paper links in json file
    '''
    def parse_arxiv_string(s):
        parts = s.split("|")
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        arxiv_id = parts[4].strip()
        code = parts[5].strip()
        arxiv_id = re.sub(r'v\d+', '', arxiv_id)
        return date,title,authors,arxiv_id,code

    with open(filename,"r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

        json_data = m.copy()

        for keywords,v in json_data.items():
            logging.info(f'keywords = {keywords}')
            for paper_id,contents in v.items():
                contents = str(contents)

                update_time, paper_title, paper_first_author, paper_url_field, code_url = parse_arxiv_string(contents)

                # 保持原格式
                contents = "|{}|{}|{}|{}|{}|\n".format(update_time,paper_title,paper_first_author,paper_url_field,code_url)
                json_data[keywords][paper_id] = str(contents)
                logging.info(f'paper_id = {paper_id}, contents = {contents}')

                valid_link = False if '|null|' in contents else True
                if valid_link:
                    continue
                try:
                    repo_url = (get_repo_from_hf(paper_id)
                                or find_code_repo(paper_title, paper_id, paper_first_author)
                                or get_code_link(paper_title)
                                or get_code_link(paper_id))

                    if repo_url is not None:
                        new_cont = contents.replace('|null|',f'|**[link]({repo_url})**|')
                        logging.info(f'ID = {paper_id}, contents = {new_cont}')
                        json_data[keywords][paper_id] = str(new_cont)

                except Exception as e:
                    logging.error(f"exception: {e} with id: {paper_id}")
        # dump to json file
        with open(filename,"w") as f:
            json.dump(json_data,f)

def update_json_file(filename,data_dict):
    '''
    daily update json file using data_dict
    '''
    with open(filename,"r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

    json_data = m.copy()

    # update papers in each keywords
    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]

            if keyword in json_data.keys():
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename,"w") as f:
        json.dump(json_data,f)

def json_to_md(filename,md_filename,
               task = '',
               to_web = False,
               use_title = True,
               use_tc = True,
               show_badge = True,
               use_b2t = True,
               topic_groups = None):
    """
    @param filename: str
    @param md_filename: str
    @return None
    """
    def pretty_math(s:str) -> str:
        ret = ''
        match = re.search(r"\$.*\$", s)
        if match == None:
            return s
        math_start,math_end = match.span()
        space_trail = space_leading = ''
        if s[:math_start][-1] != ' ' and '*' != s[:math_start][-1]: space_trail = ' '
        if s[math_end:][0] != ' ' and '*' != s[math_end:][0]: space_leading = ' '
        ret += s[:math_start]
        ret += f'{space_trail}${match.group()[1:-1].strip()}${space_leading}'
        ret += s[math_end:]
        return ret

    def slugify(*parts) -> str:
        text = '-'.join(str(part) for part in parts if part)
        text = text.lower().replace('&', 'and')
        text = re.sub(r'[^a-z0-9]+', '-', text)
        text = re.sub(r'-+', '-', text).strip('-')
        return text or 'section'

    def get_groups():
        if not topic_groups:
            return None
        if isinstance(topic_groups, dict):
            return list(topic_groups.items())
        return topic_groups

    def write_heading(f, level: int, title: str, anchor: str):
        f.write(f'<a id="{anchor}"></a>\n')
        f.write(f"{'#' * level} {title}\n\n")

    def write_table_header(f):
        if use_title == True :
            if to_web == False:
                f.write("|Publish Date|Title|Authors|PDF|Code|\n" + "|---|---|---|---|---|\n")
            else:
                f.write("| Publish Date | Title | Authors | PDF | Code |\n")
                f.write("|:---------|:-----------------------|:---------|:------|:------|\n")

    def write_papers(f, day_content):
        write_table_header(f)

        # sort papers by date
        day_content = sort_papers(day_content)

        for _,v in day_content.items():
            if v is not None:
                f.write(pretty_math(v)) # make latex pretty

        f.write(f"\n")

    def write_back_to_top(f):
        if use_b2t:
            f.write(f"<p align=right>(<a href=#top>back to top</a>)</p>\n\n")

    DateNow = datetime.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace('-','.')

    with open(filename,"r") as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)

    # clean README.md if daily already exist else create it
    with open(md_filename,"w+") as f:
        pass

    # write data into README.md
    with open(md_filename,"a+") as f:

        if (use_title == True) and (to_web == True):
            f.write("---\n" + "layout: default\n" + "---\n\n")

        if show_badge == True:
            f.write(f"[![Contributors][contributors-shield]][contributors-url]\n")
            f.write(f"[![Forks][forks-shield]][forks-url]\n")
            f.write(f"[![Stargazers][stars-shield]][stars-url]\n")
            f.write(f"[![Issues][issues-shield]][issues-url]\n\n")

        if use_title == False:
            f.write('<a id="top"></a>\n')

        if use_title == True:
            write_heading(f, 2, "Updated on " + DateNow, "top")
        else:
            f.write("> Updated on " + DateNow + "\n")

        f.write("> Usage instructions: [here](./docs/README.md#usage)\n\n")

        groups = get_groups()

        #Add: table of contents
        if use_tc == True:
            f.write("<details>\n")
            f.write("  <summary>Table of Contents</summary>\n")
            f.write("  <ol>\n")
            if groups:
                for group_name, topics in groups:
                    visible_topics = [
                        topic for topic in topics
                        if data.get(topic) or group_name != "Old"
                    ]
                    if not visible_topics:
                        continue
                    f.write(f"    <li><a href=#{slugify(group_name)}>{group_name}</a>\n")
                    f.write("      <ol>\n")
                    for topic in visible_topics:
                        f.write(
                            f"        <li><a href=#{slugify(group_name, topic)}>{topic}</a></li>\n"
                        )
                    f.write("      </ol>\n")
                    f.write("    </li>\n")
            else:
                for keyword in data.keys():
                    day_content = data[keyword]
                    if not day_content:
                        continue
                    kw = keyword.replace(' ','-')
                    f.write(f"    <li><a href=#{kw.lower()}>{keyword}</a></li>\n")
            f.write("  </ol>\n")
            f.write("</details>\n\n")

        if groups:
            grouped_topics = set()
            rendered_topics = set()
            for group_name, topics in groups:
                visible_topics = [
                    topic for topic in topics
                    if data.get(topic) or group_name != "Old"
                ]
                if not visible_topics:
                    continue

                write_heading(f, 2, group_name, slugify(group_name))
                for topic in visible_topics:
                    grouped_topics.add(topic)
                    day_content = data.get(topic, {})
                    write_heading(f, 3, topic, slugify(group_name, topic))
                    if not day_content:
                        f.write("_No papers yet._\n\n")
                        continue
                    write_papers(f, day_content)
                    rendered_topics.add(topic)
                    write_back_to_top(f)

            for keyword in data.keys():
                if keyword in grouped_topics or keyword in rendered_topics:
                    continue
                day_content = data[keyword]
                if not day_content:
                    continue
                write_heading(f, 2, keyword, slugify(keyword))
                write_papers(f, day_content)
                write_back_to_top(f)
        else:
            for keyword in data.keys():
                day_content = data[keyword]
                if not day_content:
                    continue
                # the head of each part
                f.write(f"## {keyword}\n\n")
                write_papers(f, day_content)
                write_back_to_top(f)

        if show_badge == True:
            # we don't like long string, break it!
            f.write((f"[contributors-shield]: https://img.shields.io/github/"
                     f"contributors/Vincentqyw/cv-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[contributors-url]: https://github.com/Vincentqyw/"
                     f"cv-arxiv-daily/graphs/contributors\n"))
            f.write((f"[forks-shield]: https://img.shields.io/github/forks/Vincentqyw/"
                     f"cv-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[forks-url]: https://github.com/Vincentqyw/"
                     f"cv-arxiv-daily/network/members\n"))
            f.write((f"[stars-shield]: https://img.shields.io/github/stars/Vincentqyw/"
                     f"cv-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[stars-url]: https://github.com/Vincentqyw/"
                     f"cv-arxiv-daily/stargazers\n"))
            f.write((f"[issues-shield]: https://img.shields.io/github/issues/Vincentqyw/"
                     f"cv-arxiv-daily.svg?style=for-the-badge\n"))
            f.write((f"[issues-url]: https://github.com/Vincentqyw/"
                     f"cv-arxiv-daily/issues\n\n"))

    logging.info(f"{task} finished")

def demo(**config):
    data_collector = []
    data_collector_web= []

    keywords = config['kv']
    max_results = config['max_results']
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']
    publish_wechat = config['publish_wechat']
    show_badge = config['show_badge']
    topic_groups = config.get('topic_groups')

    b_update = config['update_paper_links']
    logging.info(f'Update Paper Link = {b_update}')
    if config['update_paper_links'] == False:
        logging.info(f"GET daily papers begin")
        for topic, keyword in keywords.items():
            logging.info(f"Keyword: {topic}")
            data, data_web = get_daily_papers(topic, query = keyword,
                                            max_results = max_results)
            data_collector.append(data)
            data_collector_web.append(data_web)
            print("\n")
        logging.info(f"GET daily papers end")

    # 1. update README.md file
    if publish_readme:
        json_file = config['json_readme_path']
        md_file   = config['md_readme_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file,data_collector)
        json_to_md(json_file,md_file, task ='Update Readme',
                   show_badge = show_badge, topic_groups = topic_groups)

    # 2. update docs/index.md file (to gitpage)
    if publish_gitpage:
        json_file = config['json_gitpage_path']
        md_file   = config['md_gitpage_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file,data_collector)
        json_to_md(json_file, md_file, task ='Update GitPage',
                   to_web = True, show_badge = show_badge,
                   use_tc=False, use_b2t=False,
                   topic_groups = topic_groups)

    # 3. Update docs/wechat.md file
    if publish_wechat:
        json_file = config['json_wechat_path']
        md_file   = config['md_wechat_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector_web)
        json_to_md(json_file, md_file, task ='Update Wechat',
                   to_web=False, use_title= False,
                   show_badge = show_badge,
                   topic_groups = topic_groups)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',type=str, default='config.yaml',
                            help='configuration file path')
    parser.add_argument('--update_paper_links', default=False,
                        action="store_true",help='whether to update paper links etc.')
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = {**config, 'update_paper_links':args.update_paper_links}
    demo(**config)

    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "commit"], check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        print("Git commands executed successfully.")
    except subprocess.CalledProcessError as e:
        pass
