from flask import Flask, render_template, request
import pickle

# Read in saved model and files needed for making recommendations
model = pickle.load(open('static/model/final_model.pkl', 'rb'))  # opening pickle file in read mode
user_map = pickle.load(open('static/model/user_map.pkl', 'rb'))
user_ids = pickle.load(open('static/model/user_ids.pkl', 'rb'))
item_map = pickle.load(open('static/model/item_map.pkl', 'rb'))
item_ids = pickle.load(open('static/model/item_ids.pkl', 'rb'))
bm25_coo_train  = pickle.load(open('static/model/bm25_coo_train.pkl', 'rb'))

# initializing Flask app
app = Flask(__name__) 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend_for_user', methods=['POST'])
def recommend_for_user():
    customer_id = request.form.get("customer_id")
    text1 = 'The recommended articless are:'
    item_list = _recommend_for_user(customer_id)
    return render_template("index.html", text1=text1, item_list=item_list)

@app.route('/recommend_for_item', methods=['POST'])
def recommend_for_item():
    article_id = request.form.get("article_id")
    article_num = request.form.get("article_num")
    text2 = f'You asked for {article_id}.'
    text3 = 'You may also like these articles:'
    articles = _recommend_for_item(article_id, article_num)
    return render_template("index.html", text2=text2, input_article=article_id, text3=text3, articles=articles)

def _recommend_for_user(customer_id):
    """Given a customer ID, generate a list of items the customer may want to buy.
    A sample customer_id is '00000dbacae5abe5e23885899a1fa44253a17956c6d1c3d25f88aa139fdfc657'

    Args:
        customer_id (str): customer id
    Returns:
        a list of articles
    """
    # Get user id
    user_id = user_map[customer_id]
    
    # get 12 items for this customer
    recommend_ids, scores = model.recommend(user_id, bm25_coo_train[user_id], N=12, filter_already_liked_items=False)
    article_ids = [item_ids[item_id] for item_id in recommend_ids]
    return article_ids
    
def _recommend_for_item(article_id, article_num):
    """Given an article ID, generate a list of article_num similar items.
    A sample article_id is '0778064038'

    Args:
        articler_id (str): article id
    Returns:
        a list of articles
    """
    # Get item_id
    item_id = item_map[article_id]

    # get article_num items similar to user input article
    ids, scores= model.similar_items(itemid=item_id, N=int(article_num)+1)
    articles = [item_ids[id] for id in ids]
    return articles[1:]

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8000, debug=True)
    app.run(debug=True)