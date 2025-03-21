
import pickle  # Thư viện để lưu và tải đối tượng Python
from tensorflow.keras.models import load_model  # Để tải mô hình đã huấn luyện
from tensorflow.keras.preprocessing import sequence, text  # Để xử lý văn bản và chuỗi
import numpy as np  # Thư viện cho các phép toán số học
from pyvi import ViTokenizer  # Thư viện để phân tách từ trong tiếng Việt
import re
with open('./tokenizer/tokenizer.pickle', 'rb') as handle:  # Mở tệp chứa tokenizer đã lưu
    tokenizer = pickle.load(handle)  # Tải tokenizer từ tệp

model = load_model('D:\AI\models\V3.Text_CNN_model_len550_e100_filter64.h5')  # Tải mô hình đã huấn luyện từ tệp

sequence_length = 550  # Độ dài chuỗi đã sử dụng trong quá trình huấn luyện

# ------------------ Preprocessing Function ------------------ #
STOPWORDS = './vietnamese/vietnamese-stopwords-dash.txt'  # Đường dẫn đến tệp stopwords

with open(STOPWORDS, "r") as ins:  # Mở tệp stopwords
    stopwords = set(line.strip('\n') for line in ins)  # Đọc stopwords vào một tập hợp

def filter_stop_words(train_sentences, stop_words):
    new_sent = [word for word in train_sentences.split() if word not in stop_words]  # Lọc bỏ stopwords
    train_sentences = ' '.join(new_sent)  # Ghép lại thành chuỗi
    return train_sentences  # Trả về chuỗi đã lọc

def deEmojify(text):
    regrex_pattern = re.compile(pattern="["  # Biểu thức chính quy để loại bỏ emoji
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)  # Thay thế emoji bằng chuỗi rỗng

def preprocess_input(text):
    """Preprocesses input text for prediction."""
    text = ViTokenizer.tokenize(text)  # Phân tách từ bằng pyvi
    text = filter_stop_words(text, stopwords)  # Lọc bỏ stopwords
    text = deEmojify(text)  # Loại bỏ emoji
    text = text.lower()  # Chuyển về chữ thường
    return text  # Trả về văn bản đã được tiền xử lý

# ------------------ Prediction Function ------------------ #
def predict_text(text):
    """Predicts the label of the given text."""
    
    processed_text = preprocess_input(text)  # Tiền xử lý văn bản đầu vào

    text_sequence = tokenizer.texts_to_sequences([processed_text])  # Chuyển đổi văn bản thành chuỗi số
    text_padded = sequence.pad_sequences(text_sequence, maxlen=sequence_length)  # Padding để đảm bảo độ dài chuỗi

    prediction = model.predict(text_padded)  # Dự đoán nhãn cho chuỗi đã được padding

    predicted_label = np.argmax(prediction)  # Lấy nhãn dự đoán (chỉ số có xác suất cao nhất)

    label_mapping = {0: "clean", 1: "offensive", 2: "hate"}  # Bản đồ nhãn dự đoán đến ý nghĩa của chúng
    predicted_class = label_mapping.get(predicted_label, "unknown")  # Lấy ý nghĩa của nhãn dự đoán

    return predicted_class  # Trả về lớp dự đoán

# ------------------ Example Usage ------------------ #
input_text = "Đề nghị 17, 22,23,32 đến phà vài hơi vào mẹt tk này để nó đi cách ly thế giới hẳn đi."  # Văn bản đầu vào khác
input_text = "tôi yêu bạn"  # Văn bản đầu vào mẫu

predicted_class = predict_text(input_text)  # Dự đoán lớp cho văn bản đầu vào
print(f"Predicted class for '{input_text}'\n ==> {predicted_class}")  # In ra lớp dự đoán