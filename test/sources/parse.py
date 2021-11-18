import json, time
from lxml import etree


def to_datetime(int_time):
    '''
    时间戳转成标准格式的时间
    :param int_time:
    :return:
    '''
    int_time = int(int_time)
    date_time = time.localtime(int_time)
    date_time = time.strftime('%Y-%m-%d %H:%M:%S', date_time)
    return date_time


def get_xpath(selector, xpath, is_list=True):
    '''
    通过传入的xpath语法获取内容
    :param selector: etree.HTML(html)
    :param xpath: xpath语法
    :param is_list: 返回值是否为list
    :return: value
    '''
    value = selector.xpath(xpath)
    if not is_list:
        if not value:
            value = ''
        else:
            value = value[0]
    return value


class DetailParser():
    source = "detail"
    field_list = ['user_id', 'crawl_time', 'url_token', 'user_name', 'is_vip', 'is_verify', 'is_excellent_answer', "verify_info", "excellent_answer_info", 'update_time', 'answer_count', 'video_count',
                  'question_count', 'articles_count',
                  'columns_count', 'idea_count', 'saved_count', 'voted_up', 'follow_count', 'follower_count', 'support_live_count', 'follow_topic_count', 'follow_columns_count',
                  'follow_question_count', 'follow_favorites_list_count']

    def parse_line(self, line):
        data = json.loads(line)
        # 判断类型
        source = ""
        if data.get("activities"):
            source = "detail"
        url_token = list(data["users"].keys())
        url_token = [i for i in url_token if "-" in i or len(i) != 32]
        if url_token:
            url_token = url_token[0]
        else:
            return False
        user_info = data["users"][url_token]
        if not user_info.get("id"):
            return False
        is_vip = user_info.get("vipInfo", {}).get("isVip", False)
        badge = user_info.get("badge", [])
        identity = [i for i in badge if i["type"] == "identity"]
        best_answerer = [i for i in badge if i["type"] == "best_answerer"]
        is_verify = False
        verify_info = ""
        if identity:
            is_verify = True
            verify_info = identity[0]["description"]

        is_excellent_answer = False
        excellent_answer_info = ""
        if best_answerer:
            is_excellent_answer = True
            excellent_answer_info = ";".join([i["name"] for i in best_answerer[0].get("topics", [])])

        activities = data["activities"]
        update_time = [int(v["createdTime"]) for i, v in activities.items()]
        if update_time:
            update_time = to_datetime(max(update_time))
        else:
            update_time = ""

        save_data = {
            "user_id": user_info["id"],
            "crawl_time": data["crawl_time"],
            "url_token": user_info["urlToken"],
            "user_name": user_info["name"],
            "is_vip": is_vip,
            "is_verify": is_verify,  # 是否认证
            "is_excellent_answer": is_excellent_answer,  # 是否优秀答主
            "verify_info": verify_info,
            "excellent_answer_info": excellent_answer_info,
            "update_time": update_time,
            "answer_count": user_info.get("answerCount", ""),
            "video_count": user_info.get("zvideoCount", ""),
            "question_count": user_info.get("questionCount", ""),
            "articles_count": user_info.get("articlesCount", ""),
            "columns_count": user_info.get("columnsCount", ""),  # 专栏数
            "idea_count": user_info.get("pinsCount", ""),  # 想法数
            "saved_count": user_info.get("favoriteCount", ""),  # 收藏数
            "voted_up": user_info.get("voteupCount", ""),  # 获赞数
            "follow_count": user_info.get("followingCount", ""),  # 关注数
            "follower_count": user_info.get("followerCount", ""),  # 粉丝数
            "support_live_count": user_info.get("participatedLiveCount", ""),  # 赞助的live数
            "follow_topic_count": user_info.get("followingTopicCount", ""),  # 关注的话题数
            "follow_columns_count": user_info.get("followingColumnsCount", ""),  # 关注的专栏数
            "follow_question_count": user_info.get("followingQuestionCount", ""),  # 关注的问题数
            "follow_favorites_list_count": user_info.get("followingFavlistsCount", ""),  # 关注的收藏夹数
        }
        data_list = [save_data[field] for field in self.field_list]
        return data_list


class AnswerParser():
    source = "answer"
    field_list = ["id", "crawl_time", "user_id", "url_token", "user_name", "question_title", "question_id", "answer_id", "voteup_count", "comment_count", "updated_time", "ad_count",
                  "mcn_link_card_count", "link_card_count", "ad_link_card_count", "href_count", "mcn_link_card_list", "link_card_list", "ad_link_card_list", "content"]

    def parse_line(self, line):
        data = json.loads(line)
        # 判断类型
        source = ""
        if data.get("type", "") == "answer":
            source = "answer"
        content = data["content"]
        ad_ele_list = []
        mcn_link_card_list = []
        link_card_list = []
        ad_card_list = []
        href_list = []
        if content:
            selector = etree.HTML(content)
            if selector is not None:
                # 判断广告类型
                ad_ele_list = get_xpath(selector, "//a[@data-draft-type]")
                for ele in ad_ele_list:
                    data_type = get_xpath(ele, "./@data-draft-type", False)
                    if data_type == "mcn-link-card":
                        mcn_link_card_list.append(get_xpath(ele, "./@data-mcn-id", False))
                    if data_type == "link-card":
                        link_card_list.append(get_xpath(ele, "./@href", False))
                    if data_type == "ad-link-card":
                        ad_card_list.append(get_xpath(ele, "./@data-ad-id", False))
                href_list = get_xpath(selector, "//@href")

        save_data = {
            "id": data["id"],
            "crawl_time": data["crawl_time"],
            "user_id": data["author"]["id"],
            "url_token": data["url_token"],
            "user_name": data["author"]["name"],
            "question_title": data["question"]["title"],
            "question_id": data["question"]["id"],
            "answer_id": data["id"],
            "voteup_count": data["voteup_count"],
            "comment_count": data["comment_count"],
            "updated_time": data["updated_time"],
            "ad_count": len(ad_ele_list),
            "mcn_link_card_count": len(mcn_link_card_list),
            "link_card_count": len(link_card_list),
            "ad_link_card_count": len(ad_card_list),
            "href_count": len(href_list),
            "mcn_link_card_list": json.dumps(mcn_link_card_list),
            "link_card_list": json.dumps(link_card_list),
            "ad_link_card_list": json.dumps(ad_card_list),
            "content": content
        }
        data_list = [save_data[field] for field in self.field_list]
        return data_list


class ArticleParser():
    source = "article"
    field_list = ["id", "crawl_time", "user_id", "url_token", "user_name", "title", "article_id", "voteup_count", "comment_count", "updated_time", "ad_count",
                  "mcn_link_card_count", "link_card_count", "ad_link_card_count", "href_count", "mcn_link_card_list", "link_card_list", "ad_link_card_list", "content"]

    def parse_line(self, line):
        # 判断类型
        data = json.loads(line)
        source = ""
        if data.get("type", "") == "article":
            source = "article"
        content = data["content"]
        ad_ele_list = []
        mcn_link_card_list = []
        link_card_list = []
        ad_card_list = []
        href_list = []
        if content:
            selector = etree.HTML(content)
            if selector is not None:
                # 判断广告类型
                ad_ele_list = get_xpath(selector, "//a[@data-draft-type]")
                for ele in ad_ele_list:
                    data_type = get_xpath(ele, "./@data-draft-type", False)
                    if data_type == "mcn-link-card":
                        mcn_link_card_list.append(get_xpath(ele, "./@data-mcn-id", False))
                    if data_type == "link-card":
                        link_card_list.append(get_xpath(ele, "./@href", False))
                    if data_type == "ad-link-card":
                        ad_card_list.append(get_xpath(ele, "./@data-ad-id", False))
                href_list = get_xpath(selector, "//@href")

        save_data = {
            "id": data["id"],
            "crawl_time": data["crawl_time"],
            "user_id": data["author"]["id"],
            "url_token": data["url_token"],
            "user_name": data["author"]["name"],
            "title": data["title"],
            "article_id": data["id"],
            "voteup_count": data["voteup_count"],
            "comment_count": data["comment_count"],
            "updated_time": data["updated"],
            "ad_count": len(ad_ele_list),
            "mcn_link_card_count": len(mcn_link_card_list),
            "link_card_count": len(link_card_list),
            "ad_link_card_count": len(ad_card_list),
            "href_count": len(href_list),
            "mcn_link_card_list": json.dumps(mcn_link_card_list),
            "link_card_list": json.dumps(link_card_list),
            "ad_link_card_list": json.dumps(ad_card_list),
            "content": content
        }
        data_list = [save_data[field] for field in self.field_list]
        return data_list
