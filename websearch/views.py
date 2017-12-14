from django.shortcuts import render
from django.utils.safestring import mark_safe
from whoosh.qparser import QueryParser
from django.views.decorators.csrf import csrf_protect
import datetime
import os
import re
import time
from queue import Queue
from urllib import request
from urllib.error import HTTPError
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from numpy import transpose, zeros, dot
import shutil
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in, open_dir
from jieba.analyse import ChineseAnalyzer


def is_useless(urls, filter_true, filter_false):  # 无效链接
    if urls is None:
        return True
    if filter_true.findall(urls):
        return True
    if not filter_false.findall(urls):
        return True
    return False


def start_spider(num):
    init_url = "http://www.nankai.edu.cn/"  # 用于拼接网页

    html_num = 1

    path_for_html = './html'
    path_for_urls = './urls'
    path_for_index = './index.txt'

    reg_type = r'\.js|\.css|\.png|\.gif|\.pdf|\.jpg|download|Download|\.mp4|\.mp3|\.doc|\.rar|\.zip|\.bmp|\.apk'
    filter_true = re.compile(reg_type)  # 这个地方参照吴大大
    reg_sites = r'(nankai.edu.cn|222.30|202.113)'
    filter_false = re.compile(reg_sites)

    already_seen = set()  # 用于记录已经爬取过的或者正在等待爬取网页
    urls_que = Queue()  # 用于缓存等待爬取的网页
    urls_que.put(init_url)
    already_seen.add(init_url)

    if not os.path.isdir(path_for_html):  # 创建爬虫文件保存的地方
        os.mkdir(path_for_html)
    else:
        shutil.rmtree(path_for_html)
        os.mkdir(path_for_html)
    if not os.path.isdir(path_for_urls):
        os.mkdir(path_for_urls)
    else:
        shutil.rmtree(path_for_urls)
        os.mkdir(path_for_urls)

    if os.path.exists(path_for_index):  # 如果存在就先删掉
        os.remove(path_for_index)

    while not urls_que.empty() and html_num <= num:  # 开始爬虫，第一个参数是传入要爬的网页数
        print(html_num)
        url = urls_que.get()
        try:
            response = requests.get(url, timeout=2)  # 抓取页面
            response.encoding = response.apparent_encoding  # 修改页面编码

            path = path_for_html + '/' + str(html_num) + '.html'  # 加载页面生成网页快照
            request.urlretrieve(url, path)

            path = path_for_urls + '/' + str(html_num) + '.txt'  # 记录锚文本
            file = open(path, 'w')
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a'):  # 找到所有的a标签
                urls = link.get('href')  # 获取链接
                url_read = urljoin(init_url, urls)  # 拼接链接
                if not is_useless(urls, filter_true, filter_false):
                    if url_read not in already_seen:  # 如果还没爬到过这个网页就添加
                        urls_que.put(url_read)
                        already_seen.add(url_read)
                    try:
                        file.write(url_read + ' ' + link.get_text() + '\n')
                    except UnicodeEncodeError:
                        print('UnicodeEncode wrong when writing urls...\n')
                        pass
            file.close()

            try:  # 写入索引文件
                if os.path.exists(path_for_index):
                    file = open(path_for_index, 'a')
                else:
                    file = open(path_for_index, 'w')
                file.write(str(html_num) + ' ' + url + '\n')
            except UnicodeEncodeError:
                print('UnicodeEncode wrong when writing index.txt...\n')
                pass
            finally:
                file.close()
            html_num += 1  # 自增

        except requests.exceptions.HTTPError:
            print(response.status_code, " :HTTPError", " ", url)
            pass
        except requests.exceptions.ConnectionError:
            print(response.status_code, " :ConnectionError", " ", url)
            pass
        except requests.exceptions.ReadTimeout:
            print(response.status_code, " :ReadTimeout", " ", url)
            pass
        except HTTPError:
            print(response.status_code, " :HTTPError", " ", url)
            pass
        except requests.exceptions.InvalidSchema:
            print(response.status_code, " :InvalidSchema", " ", url)
            pass
        except:
            print(response.status_code, " :Unknown", " ", url)
            pass
        time.sleep(0.5)


def start_page_rank():
    a = get_matrix()
    M = graph_move(a)
    pr = first_pr(M)
    ans = page_rank(0.85, M, pr)
    with open("./pagerank.txt", "w") as file:
        for i in range(ans.shape[0]):
            for j in range(ans.shape[1]):
                file.write(str(ans[i][j])+'\n')
    print(ans)


def get_matrix():
    urls = get_index()
    path = "./urls"  # 获取当前目录
    files = os.listdir(path)  # 获取目录下所有的文件名
    length = len(files)  # 获取当前文件数并给a分配空间
    a = zeros((length, length), dtype=float)
    for filename in files:
        with open(path + "/" + filename, "r") as file:
            line = file.readline()
            while line:
                try:
                    des = urls[line.split(' ')[0]]
                    a[int(filename.split('.')[0])][int(des)] += 1
                except IndexError:
                    pass
                except KeyError:
                    pass
                finally:
                    line = file.readline()
    return a


def get_index():
    urls = {}
    with open("./index.txt", "r") as file:
        line = file.readline()
        while line:
            try:
                url = line.split(' ')[1].split('\n')[0]
                urls[url] = line.split(' ')[0]
                line = file.readline()
            except IndexError:
                print("Out of index...")
                pass
    return urls


def graph_move(a):  # 构造转移矩阵
    b = transpose(a)  # b为a的转置矩阵
    c = zeros((a.shape), dtype=float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if b[j].sum() != 0:
                c[i][j] = a[i][j] / (b[j].sum())  # 完成初始化分配
    return c


def first_pr(c):  # pr值得初始化
    pr = zeros((c.shape[0], 1), dtype=float)  # 构造一个存放pr值得矩阵
    for i in range(c.shape[0]):
        pr[i] = float(1) / c.shape[0]
    return pr


def page_rank(p, m, v):  # 计算pageRank值
    # 判断pr矩阵是否收敛,(v == p*dot(m,v) + (1-p)*v).all()判断前后的pr矩阵是否相等，若相等则停止循环
    while not (v == p * dot(m, v) + (1 - p) * v).all():
        v = p * dot(m, v) + (1 - p) * v
    return v


@csrf_protect
def start(request):
    num = request.POST.get("num")
    if num is not None:
        num = int(num)
        start_spider(num)
        start_page_rank()
    return render(request, 'index.html')


@csrf_protect
def analyse(request):
    ini_index()
    return render(request, 'index.html')


def read_html(writer):
    root = "./html"
    files = os.listdir(root)  # 获取目录下所有的文件名
    for filename in files:
        with open(root + "/" + filename, "rb") as html:
            soup = BeautifulSoup(html, "html.parser")
            try:
                title_read = soup.find("title").get_text()
            except AttributeError:
                title_read = ""
            [s.extract() for s in soup('script')]  # 去除script和style标签
            [s.extract() for s in soup('style')]
            content_read = soup.get_text()
            writer.add_document(
                title=title_read,
                content=content_read,
                path=root + "/" + filename)


def ini_index():
    analyzer = ChineseAnalyzer()
    schema = Schema(
        title=TEXT(stored=True),
        content=TEXT(stored=True, analyzer=analyzer),
        path=ID(stored=True), )
    if not os.path.exists("index"):
        os.mkdir("index")
    ix = create_in("index", schema)
    writer = ix.writer()
    read_html(writer)  # 获取title content等并建立索引
    writer.commit()


@csrf_protect
def search(request):
    querystring = request.POST.get("query")
    if 'HTTP_X_FORWARDED_FOR' in request.META:
        ip = request.META['HTTP_X_FORWARDED_FOR']
    else:
        ip = request.META['REMOTE_ADDR']
    now = datetime.datetime.now()
    with open("./log.txt", "a") as log:
        log.write("from: " + ip + " time: " + now.strftime('%Y-%m-%d %H:%M:%S')
                  + " query: " + querystring + '\n')
    ix = open_dir("./index")
    hits = []
    with ix.searcher() as searcher:
        myquery = QueryParser("content", ix.schema).parse(querystring)
        results = searcher.search(myquery)
        for hit in results:
            content = mark_safe(hit.highlights("content"))
            hits.append((hit["title"], content, hit["path"]))
        # for hit in results:
        #     content = mark_safe(hit.highlights("content"))
        #     hits.append((hit["title"], content, hit["url"]))
        return render(request, 'result.html', {'results': hits})
