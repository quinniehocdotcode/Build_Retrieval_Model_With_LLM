import nvtabular as nvt
from nvtabular.ops import Categorify, Normalize, FillMissing, Rename
import os

print("Bắt đầu quy trình xử lý dữ liệu bằng NVTabular...")

# --- 1. Định nghĩa các cột và đường dẫn ---
INPUT_DATA_PATH = 'raw_customer_data.csv'
OUTPUT_PATH = 'processed_data'
WORKFLOW_PATH = 'workflow'

# Phân loại các cột đặc trưng
# 'job_title' sẽ được xử lý bởi LLM trong pipeline đầy đủ, ở đây ta tạm coi nó là categorical
categorical_features = ['location', 'gender', 'job_title']
continuous_features = ['age', 'income_monthly']
# user_id là cột định danh, chúng ta muốn giữ lại nó
id_feature = ['user_id']

# --- 2. Định nghĩa quy trình xử lý (Workflow) ---

# Luồng xử lý cho các đặc trưng số (continuous):
# 1. Điền các giá trị bị thiếu bằng giá trị trung bình của cột.
# 2. Chuẩn hóa các giá trị về khoảng [0, 1].
# 3. Đổi tên cột để rõ ràng hơn (ví dụ: 'age' -> 'age_norm').

cont_workflow = (
    continuous_features
    >> FillMissing(fill_val=0)  # Có thể dùng strategy='mean' nhưng fill_val=0 đơn giản hơn
    >> Normalize()
    >> Rename(postfix='_norm')
)

# Luồng xử lý cho các đặc trưng danh mục (categorical):
# 1. Biến đổi các chuỗi thành chỉ số số nguyên.
# 2. Đổi tên cột để rõ ràng hơn (ví dụ: 'location' -> 'location_cat').
cat_workflow = (
    categorical_features
    >> Categorify()
    >> Rename(postfix='_cat')
)

# Kết hợp tất cả các luồng xử lý lại thành một workflow hoàn chỉnh
# Chúng ta muốn giữ lại cột 'user_id' và thêm các cột đã xử lý
output_features = id_feature + cont_workflow + cat_workflow
workflow = nvt.Workflow(output_features)

# --- 3. Áp dụng Workflow vào dữ liệu ---

# Tạo đối tượng Dataset của NVTabular.
# Nó có thể xử lý các file lớn hơn bộ nhớ RAM bằng cách đọc theo từng chunk.
# engine='csv' chỉ định rằng chúng ta đang đọc từ file CSV.
dataset = nvt.Dataset(INPUT_DATA_PATH, engine='csv')

# 'fit' workflow vào dataset: NVTabular sẽ thực hiện một lượt quét dữ liệu
# để học các từ điển (cho Categorify) và các giá trị thống kê (cho Normalize).
print("\nBắt đầu bước 'fit' để học từ điển và thống kê...")
workflow.fit(dataset)
print("Bước 'fit' hoàn tất.")

# 'transform' dataset: Áp dụng các phép biến đổi đã học vào dữ liệu
# và lưu kết quả đầu ra.
print("\nBắt đầu bước 'transform' để xử lý và lưu dữ liệu...")
processed_dataset = workflow.transform(dataset)

# Lưu dữ liệu đã xử lý dưới định dạng Parquet (hiệu quả hơn CSV)
# và lưu workflow đã học.
processed_dataset.to_parquet(output_path=OUTPUT_PATH)
workflow.save(WORKFLOW_PATH)

print(f"\nQuy trình hoàn tất!")
print(f"  - Dữ liệu đã xử lý được lưu tại thư mục: '{OUTPUT_PATH}'")
print(f"  - Workflow (chứa từ điển và metadata) được lưu tại thư mục: '{WORKFLOW_PATH}'")

# --- 4. (Tùy chọn) Kiểm tra kết quả ---
print("\nĐọc và hiển thị 5 dòng đầu của dữ liệu đã xử lý:")
# Đọc lại file parquet đã lưu để xem kết quả
processed_df = pd.read_parquet(OUTPUT_PATH)
print(processed_df.head())
