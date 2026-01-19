import numpy as np
from bs4 import BeautifulSoup

# 包含所有可能的HTML标签的列表
all_html_tags = [
    'html', 'head', 'title', 'base', 'link', 'meta', 'style', 'script', 'noscript',
    'body', 'section', 'nav', 'article', 'aside', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'header', 'footer', 'address', 'p', 'hr', 'pre', 'blockquote', 'ol', 'ul', 'li',
    'dl', 'dt', 'dd', 'figure', 'figcaption', 'main', 'div', 'a', 'em', 'strong', 'small',
    's', 'cite', 'q', 'dfn', 'abbr', 'data', 'time', 'code', 'var', 'samp', 'kbd', 'sub',
    'sup', 'i', 'b', 'u', 'mark', 'ruby', 'rt', 'rp', 'bdi', 'bdo', 'span', 'br', 'wbr',
    'ins', 'del', 'picture', 'source', 'img', 'iframe', 'embed', 'object', 'param',
    'video', 'audio', 'track', 'map', 'area', 'table', 'caption', 'colgroup', 'col',
    'tbody', 'thead', 'tfoot', 'tr', 'td', 'th', 'form', 'fieldset', 'legend', 'label',
    'input', 'button', 'select', 'datalist', 'optgroup', 'option', 'textarea', 'output',
    'progress', 'meter', 'details', 'summary', 'menu', 'menuitem', 'dialog', 'script',
    'noscript', 'template', 'canvas'
]


html_tags = [
    'html', 'head', 'title', 'link', 'meta', 'style', 'script', 'body', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'header', 'footer', 'address', 'ul', 'li', 'div', 'a', 'span', 'p',
    'picture', 'source', 'img', 'map', 'area', 'table', 'form', 'legend', 'label',
    'input', 'button', 'select', 'option', 'textarea', 'menu', 'menuitem'
]

# 创建一个标签到索引的映射
tag_to_index = {tag: idx for idx, tag in enumerate(html_tags)}


def get_depth(tag, result, count, depth=0):
    if tag.name in html_tags:
        index = tag_to_index[tag.name]
        result[index] += depth
        count[index] += 1
    else:
        # print(tag.name)
        result[len(result)-1] += depth
        count[len(count)-1] += 1
    for child in tag.children:
        if child.name:
            get_depth(child, result, count, depth + 1)

def calculate_average_depth(result, count):
    average_depth = [0] * len(result)
    for i in range(len(result)):
        if count[i] > 0:
            average_depth[i] = result[i] / count[i]
    return average_depth

def get_state_embedding(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # print(soup.prettify())
    all_tags = soup.find_all()
    depth_tag = [0] * (len(html_tags) + 1)
    count_tag = [0] * (len(html_tags) + 1)
    for tag in all_tags:
        get_depth(tag, depth_tag, count_tag)

    avera_depth = calculate_average_depth(depth_tag, count_tag)
    if isinstance(avera_depth, list):
        avera_depth = np.array(avera_depth)
    return avera_depth


# if __name__ == '__main__':
#     html_content = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <title>Example Page</title>
#     </head>
#     <body>
#         <div id="content">
#             <h1>Welcome to the Example Page</h1>
#             <p>This is a simple example.</p>
#             <a href="https://www.example.com">Visit Example.com</a>
#         </div>
#         <div id="sidebar">
#             <h2>Sidebar</h2>
#             <ul>
#                 <li>Item 1</li>
#                 <li>Item 2</li>
#             </ul>
#         </div>
#     </body>
#     </html>
#     """
#
#     answer = get_state_embedding(html_content)
#     print(answer)