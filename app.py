# --- FILE: app.py ---
# (Đã sửa lỗi gợi ý sai bằng Stop Words)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import json

# --- PHẦN 1: "BỘ NÃO" TF-IDF ---

def load_data_and_model():
    """Hàm tải file và huấn luyện mô hình TF-IDF."""
    print("--- [Bộ não Sản phẩm] Đang tải dữ liệu và huấn luyện mô hình... ---")
    try:
        df = pd.read_csv('sang.csv', sep=';')
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file 'sang.csv'.")
        return None, None, None
    
    # Xử lý dữ liệu
    feature_columns = [
        'category', 'temperature', 'caffeine_level', 'dairy', 
        'mood_tag', 'season', 'tags', 'name', 'description_vn'
    ]
    for col in feature_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('')
    
    df['features'] = (
        df['category'] + ' ' + df['temperature'] + ' ' +
        df['caffeine_level'] + ' ' + df['dairy'] + ' ' +
        df['mood_tag'] + ' ' + df['season'] + ' ' +
        df['tags'] + ' ' + df['name'] + ' ' + df['description_vn']
    )
    
    # --- SỬA ĐỔI BẮT ĐẦU TỪ ĐÂY ---
    
    # 1. Thêm danh sách STOP WORDS (từ vô nghĩa)
    # Giúp AI tập trung vào từ khóa chính (ví dụ: "nóng", "lạnh", "tỉnh táo")
    # và bỏ qua các từ nhiễu (ví dụ: "tôi", "muốn", "uống", "gì đó")
    vietnamese_stop_words = [
        "tôi", "muốn", "uống", "gì", "đó", "một", "cái", "cho", "bạn", "và", "là", 
        "có", "không", "thì", "mà", "ở", "với", "để", "làm", "cũng", "được", 
        "của", "trên", "dưới", "hay", "vào", "ra", "lên", "này", "khi", "nào",
        "anh", "chị", "em", "chú", "cô", "bác", "xin", "cảm", "ơn", "món"
    ]

    # 2. Huấn luyện mô hình (thêm stop_words=...)
    print("--- [Bộ não Sản phẩm] Đang huấn luyện với STOP WORDS... ---")
    
    # SỬA DÒNG NÀY: Thêm (stop_words=vietnamese_stop_words)
    tfidf = TfidfVectorizer(stop_words=vietnamese_stop_words)
    
    tfidf_matrix = tfidf.fit_transform(df['features'])
    
    # --- HẾT PHẦN SỬA ĐỔI ---
    
    print("--- [Bộ não Sản phẩm] Đã sẵn sàng! ---")
    return df, tfidf, tfidf_matrix

def recommend_by_query(query, tfidf_vectorizer, product_tfidf_matrix, data_df, num_results=5):
    """Gợi ý sản phẩm dựa trên một câu mô tả tự do."""
    query_vec = tfidf_vectorizer.transform([query])
    sim_scores = cosine_similarity(query_vec, product_tfidf_matrix)
    sim_scores_list = list(enumerate(sim_scores[0]))
    sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)
    sim_scores_list = sim_scores_list[0 : num_results]
    product_indices = [i[0] for i in sim_scores_list]
    
    # Chỉ trả về các cột cần thiết
    results_df = data_df.iloc[product_indices][['name', 'description_vn', 'price_vnd']]
    
    # Chuyển DataFrame thành dạng JSON
    return json.loads(results_df.to_json(orient='records'))

# --- PHẦN 2: "API" (Flask Server) ---

app = Flask(__name__)

# Tải data và mô hình ngay khi bật server
data_df, tfidf_vec, tfidf_mat = load_data_and_model()

# --- Lối vào: Mở trang web (Frontend) ---
@app.route("/") # Khi ai đó truy cập trang chủ (/)
def home():
    """Phục vụ file index.html cho người dùng."""
    try:
        # Flask sẽ tự động tìm file "index.html" trong thư mục "templates"
        return render_template("index.html")
    except Exception as e:
        return f"Lỗi: không tìm thấy file index.html. Bạn đã tạo thư mục 'templates' và bỏ file vào đó chưa? {e}"

# --- Lối vào: "Bộ não" API (Cho web gọi) ---
@app.route("/search")
def search_endpoint():
    query = request.args.get('q')
    
    if not query:
        return jsonify({"error": "Bạn cần cung cấp query (ví dụ: ?q=cà phê)"}), 400
    
    if data_df is None:
        return jsonify({"error": "Server đang lỗi, không tải được dữ liệu"}), 500

    # Gọi "bộ não" TF-IDF
    results = recommend_by_query(query, tfidf_vec, tfidf_mat, data_df)
    
    # Trả kết quả (dạng JSON) về
    return jsonify(results)

# --- PHẦN 3: Chạy Server ---
if __name__ == '__main__':
    # Chạy trên cổng 5000
    app.run(debug=True, port=5000)