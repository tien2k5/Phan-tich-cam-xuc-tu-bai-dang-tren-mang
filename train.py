# !pip install keras
# !pip install tensorflow-gpu
# !pip install pyvi
# !pip install numpy
# !pip install pandas
# !pip install tensorflow
# !pip install keras
# !pip install fasttext

import pandas as pd  # Thư viện để xử lý dữ liệu dạng bảng
import numpy as np  # Thư viện cho các phép toán số học
import re  # Thư viện để xử lý chuỗi
import pickle  # Thư viện để lưu và tải đối tượng Python

import seaborn as sn  # Thư viện để vẽ biểu đồ
import matplotlib.pyplot as plt  # Thư viện để vẽ biểu đồ

from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D  # Các lớp của Keras để xây dựng mô hình CNN
from keras.layers import Reshape, Flatten, Dropout, Concatenate  # Các lớp khác của Keras
from keras.callbacks import ModelCheckpoint  # Để lưu mô hình tốt nhất trong quá trình huấn luyện
from keras.optimizers import Adam  # Bộ tối ưu Adam
from keras.models import Model  # Để xây dựng mô hình Keras
from keras.utils import to_categorical  # Để chuyển đổi nhãn thành dạng one-hot
from tensorflow.keras.preprocessing import text, sequence  # Để xử lý văn bản và chuỗi

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score  # Các hàm để đánh giá mô hình
from pyvi import ViTokenizer  # Thư viện để phân tách từ trong tiếng Việt

# --- 1. Load and Preprocess Data ---
DATA = '/content/vihsd/data/vihsd/train.csv'  # Đường dẫn đến tệp dữ liệu huấn luyện
DEV_DATA = '/content/vihsd/data/vihsd/dev.csv'  # Đường dẫn đến tệp dữ liệu phát triển
TEST_DATA = '/content/vihsd/data/vihsd/test.csv'  # Đường dẫn đến tệp dữ liệu kiểm tra
STOPWORDS = '/content/drive/MyDrive/Colab Notebooks/BTLTUYEN/BTLTUYEN/vietnamese-stopwords-dash.txt'  # Đường dẫn đến tệp stopwords

# Load stopwords
with open(STOPWORDS, "r") as ins:  # Mở tệp stopwords
    stopwords = set(line.strip('\n') for line in ins)  # Đọc stopwords vào một tập hợp

def filter_stop_words(train_sentences, stop_words):
    new_sent = [word for word in train_sentences.split() if word not in stop_words]  # Lọc bỏ stopwords
    train_sentences = ' '.join(new_sent)  # Ghép lại thành chuỗi
    return train_sentences

def deEmojify(text):
    regrex_pattern = re.compile(pattern="["  # Biểu thức chính quy để loại bỏ emoji
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r'', text)  # Thay thế emoji bằng chuỗi rỗng

def preprocess(text, tokenized=True, lowercased=True):
    text = ViTokenizer.tokenize(text) if tokenized else text  # Phân tách từ nếu cần
    text = filter_stop_words(text, stopwords)  # Lọc bỏ stopwords
    text = deEmojify(text)  # Loại bỏ emoji
    text = text.lower() if lowercased else text  # Chuyển về chữ thường nếu cần
    return text

def pre_process_features(X, y, tokenized=True, lowercased=True):
    X = [preprocess(str(p), tokenized=tokenized, lowercased=lowercased) for p in list(X)]  # Tiền xử lý dữ liệu
    # Remove empty strings and corresponding labels
    X_filtered, y_filtered = [], []  # Khởi tạo danh sách để lưu dữ liệu đã lọc
    for i, text in enumerate(X):
        if text:  # Kiểm tra nếu chuỗi không rỗng
            X_filtered.append(text)  # Thêm vào danh sách đã lọc
            y_filtered.append(y[i])  # Thêm nhãn tương ứng
    return X_filtered, np.array(y_filtered)  # Trả về dữ liệu đã lọc

# Read data
train_data = pd.read_csv(DATA)  # Đọc dữ liệu huấn luyện
dev_data = pd.read_csv(DEV_DATA)  # Đọc dữ liệu phát triển
test_data = pd.read_csv(TEST_DATA)  # Đọc dữ liệu kiểm tra

X_train = train_data['free_text']  # Lấy cột văn bản từ dữ liệu huấn luyện
y_train = train_data['label_id'].values  # Lấy cột nhãn từ dữ liệu huấn luyện

X_dev = dev_data['free_text']  # Lấy cột văn bản từ dữ liệu phát triển
y_dev = dev_data['label_id'].values  # Lấy cột nhãn từ dữ liệu phát triển

X_test = test_data['free_text']  # Lấy cột văn bản từ dữ liệu kiểm tra
y_test = test_data['label_id'].values  # Lấy cột nhãn từ dữ liệu kiểm tra

train_X, train_y = pre_process_features(X_train, y_train, tokenized=True, lowercased=True)  # Tiền xử lý dữ liệu huấn luyện
dev_X, dev_y = pre_process_features(X_dev, y_dev, tokenized=True, lowercased=True)  # Tiền xử lý dữ liệu phát triển
test_X, test_y = pre_process_features(X_test, y_test, tokenized=True, lowercased=True)  # Tiền xử lý dữ liệu kiểm tra

# --- 2. Load Embeddings and Create Features ---
EMBEDDING_FILE = '/content/drive/MyDrive/Colab Notebooks/BTLTUYEN/BTLTUYEN/cc.vi.300.vec'  # Đường dẫn đến tệp nhúng từ
MODEL_FILE = '/content/drive/MyDrive/Colab Notebooks/BTLTUYEN/V3.Text_CNN_model_len150_e60_filter64.h5'  # Đường dẫn để lưu mô hình

# /////////////////V1
# vocabulary_size = 10000
# sequence_length = 100
# embedding_dim = 300
# batch_size = 256
# epochs = 40
# drop = 0.5
# filter_sizes = [2, 3, 5]
# num_filters = 32

# /////////////////V2
vocabulary_size = 20000  # Kích thước từ vựng tăng lên
sequence_length = 150  # Độ dài chuỗi tăng lên
embedding_dim = 300  # Kích thước nhúng từ
batch_size = 100  # Kích thước lô tăng lên
epochs = 60  # Số lượng epoch tăng lên
drop = 0.4  # Tỷ lệ dropout điều chỉnh
filter_sizes = [3, 4, 5]  # Kích thước bộ lọc thay đổi một chút
num_filters = 64  # Số lượng bộ lọc tăng lên

# Load word embeddings
embeddings_index = {}  # Khởi tạo từ điển để lưu nhúng từ
count = 0
with open(EMBEDDING_FILE, encoding='utf8') as f:  # Mở tệp nhúng từ
    for line in f:
        count += 1
        values = line.rstrip().rsplit(' ')  # Tách từ và giá trị nhúng
        word = values[0]  # Từ
        coefs = np.asarray(values[1:], dtype='float32')  # Giá trị nhúng
        print(f'{count}: {word}')  # In ra từ và số thứ tự
        embeddings_index[word] = coefs  # Lưu vào từ điển

# Create tokenizer and embedding matrix
tokenizer = text.Tokenizer(lower=False, filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n')  # Khởi tạo tokenizer
tokenizer.fit_on_texts(train_X)  # Huấn luyện tokenizer trên dữ liệu huấn luyện
with open('/content/vihsd/tokenizer/tokenizer.pickle', 'wb') as handle:  # Lưu tokenizer vào tệp
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

word_index = tokenizer.word_index  # Lấy từ điển từ
num_words = len(word_index) + 1  # Số lượng từ trong từ điển
embedding_matrix = np.zeros((num_words, embedding_dim))  # Khởi tạo ma trận nhúng

for word, i in word_index.items():
    if i >= vocabulary_size:  # Kiểm tra nếu chỉ số từ lớn hơn kích thước từ vựng
        continue
    embedding_vector = embeddings_index.get(word)  # Lấy giá trị nhúng từ
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector  # Lưu giá trị nhúng vào ma trận

def make_featues(X, y, tokenizer, is_one_hot_label=True):
    """Chuyển đổi dữ liệu văn bản thành các đặc trưng số cho mô hình CNN."""
    X = [str(text) for text in X]  # Chuyển đổi tất cả văn bản thành chuỗi
    X = tokenizer.texts_to_sequences(X)  # Chuyển đổi văn bản thành các chuỗi số
    X = sequence.pad_sequences(X, maxlen=sequence_length)  # Padding để đảm bảo độ dài chuỗi

    if is_one_hot_label and (not isinstance(y, np.ndarray) or y.ndim == 1):
        y = to_categorical(y, num_classes=3)  # Chuyển đổi nhãn thành dạng one-hot

    return X, y  # Trả về dữ liệu đã chuyển đổi

# Prepare data
train_X, train_y = make_featues(train_X, train_y, tokenizer)  # Chuẩn bị dữ liệu huấn luyện
dev_X, dev_y = make_featues(dev_X, dev_y, tokenizer)  # Chuẩn bị dữ liệu phát triển
test_X, test_y = make_featues(test_X, test_y, tokenizer, is_one_hot_label=False)  # Chuẩn bị dữ liệu kiểm tra

# --- 3. Build and Train the CNN Model ---
inputs = Input(shape=(sequence_length,), dtype='int32')  # Đầu vào cho mô hình
embedding = Embedding(input_dim=num_words, output_dim=embedding_dim,
                      input_length=sequence_length, weights=[embedding_matrix])(inputs)  # Lớp nhúng
reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)  # Định hình lại đầu vào

# Các lớp tích chập
conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim),
                 padding='valid', kernel_initializer='normal', activation='elu')(reshape)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim),
                 padding='valid', kernel_initializer='normal', activation='elu')(reshpe)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim),
                 padding='valid', kernel_initializer='normal', activation='elu')(reshape)

# Các lớp max pooling
maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1),
                      strides=(1, 1), padding='valid')(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1),
                      strides=(1, 1), padding='valid')(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1),
                      strides=(1, 1), padding='valid')(conv_2)

# Kết hợp các tensor
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)  # Chuyển đổi thành một vector
dropout = Dropout(drop)(flatten)  # Thêm lớp dropout để giảm overfitting
output = Dense(units=3, activation='softmax')(dropout)  # Lớp đầu ra với softmax

model = Model(inputs=inputs, outputs=output)  # Tạo mô hình
model.summary()  # In ra tóm tắt mô hình

checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.keras',
                             monitor='val_acc', verbose=1, save_best_only=True, mode='auto')  # Lưu mô hình tốt nhất
adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  # Khởi tạo bộ tối ưu Adam
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])  # Biên dịch mô hình

model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(dev_X, dev_y))  # Huấn luyện mô hình
model.save(MODEL_FILE)  # Lưu mô hình

# --- 4. Evaluate the Model ---
prediction = model.predict(test_X, batch_size=batch_size, verbose=0)  # Dự đoán trên dữ liệu kiểm tra
y_pred = prediction.argmax(axis=-1)  # Lấy nhãn dự đoán

cf1 = confusion_matrix(test_y, y_pred)  # Tính ma trận nhầm lẫn
print(cf1)  # In ra ma trận nhầm lẫn

evaluation = f1_score(test_y, y_pred, average='micro')  # Tính F1-score (micro)
print("F1 - micro: " + str(evaluation))  # In ra F1-score (micro)

evaluation = f1_score(test_y, y_pred, average='macro')  # Tính F1-score (macro)
print("F1 - macro: " + str(evaluation))  # In ra F1-score (macro)

evaluation = accuracy_score(test_y, y_pred)  # Tính độ chính xác
print("Accuracy: " + str(evaluation))  # In ra độ chính xác

df_cm1 = pd.DataFrame(cf1, index=["clean", "offensive", "hate"],
                      columns=["clean", "offensive", "hate"])  # Tạo DataFrame cho ma trận nhầm lẫn
plt.clf()  # Xóa biểu đồ hiện tại
sn.heatmap(df_cm1, annot=True, cmap="Greys", fmt='g', cbar=True, annot_kws={"size": 30})  # Vẽ biểu đồ nhiệt cho ma trận nhầm lẫn
plt.show()  # Hiển thị biểu đồ