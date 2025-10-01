import pandas as pd
from faker import Faker
import numpy as np
import os

# Khởi tạo Faker để tạo dữ liệu giả
fake = Faker('vi_VN') # Sử dụng ngôn ngữ tiếng Việt

# --- Cấu hình tạo dữ liệu ---
NUM_CUSTOMERS = 20000 # Tăng số lượng khách hàng để dữ liệu phong phú hơn

print(f"Bắt đầu tạo dữ liệu giả cho {NUM_CUSTOMERS} khách hàng...")

# --- Định nghĩa các giá trị mẫu ---

# 1. Thuộc tính cá nhân
locations = ['Hà Nội', 'TP. HCM', 'Đà Nẵng', 'Hải Phòng', 'Cần Thơ', 'Bình Dương', 'Đồng Nai']
genders = ['Nam', 'Nữ']
job_titles = [
    'Nhân viên văn phòng', 'Kỹ sư phần mềm', 'Bác sĩ', 'Giáo viên', 
    'Giám đốc kinh doanh', 'Lao động tự do', 'Nội trợ', 'Sinh viên', 'Công nhân'
]
marital_statuses = ['Độc thân', 'Đã kết hôn', 'Ly hôn']

# 2. Thuộc tính sản phẩm bảo hiểm
insurance_products = {
    'P01': 'Bảo hiểm Sức khỏe Toàn diện',
    'P02': 'Bảo hiểm Bệnh hiểm nghèo',
    'P03': 'Bảo hiểm Nhân thọ Tích lũy',
    'P04': 'Bảo hiểm Liên kết Đầu tư',
    'P05': 'Bảo hiểm Hưu trí An nhàn',
    'P06': 'Bảo hiểm Tai nạn Cá nhân',
    'P07': 'Bảo hiểm Sức khỏe cho Gia đình'
}
product_ids = list(insurance_products.keys())

# --- Tạo DataFrame ban đầu với các thuộc tính cá nhân ---
data = {
    'user_id': range(1, NUM_CUSTOMERS + 1),
    'full_name': [fake.name() for _ in range(NUM_CUSTOMERS)],
    'age': np.random.randint(18, 65, size=NUM_CUSTOMERS),
    'location': np.random.choice(locations, size=NUM_CUSTOMERS, p=[0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1]),
    'gender': np.random.choice(genders, size=NUM_CUSTOMERS, p=[0.5, 0.5]),
    'job_title': np.random.choice(job_titles, size=NUM_CUSTOMERS),
    'marital_status': np.random.choice(marital_statuses, size=NUM_CUSTOMERS, p=[0.4, 0.5, 0.1]),
    'income_monthly': np.random.randint(7, 150, size=NUM_CUSTOMERS) * 1_000_000,
}
df = pd.DataFrame(data)

# Thêm cột 'num_children' dựa trên tình trạng hôn nhân và tuổi
df['num_children'] = 0
married_and_older = (df['marital_status'] == 'Đã kết hôn') & (df['age'] > 25)
df.loc[married_and_older, 'num_children'] = np.random.randint(1, 4, size=married_and_older.sum())

# --- Gán sản phẩm bảo hiểm đã mua (1 sản phẩm/người) ---
# Logic gán sản phẩm thông minh hơn một chút
# Tạo một cột xác suất mặc định cho mỗi sản phẩm
base_probs = [0.2, 0.2, 0.15, 0.15, 0.1, 0.15, 0.05]
purchased_products = []

for index, row in df.iterrows():
    probs = base_probs.copy()
    # Người lớn tuổi có khả năng mua bảo hiểm hưu trí cao hơn
    if row['age'] > 50:
        probs[4] += 0.2 # Index 4 là 'Bảo hiểm Hưu trí'
    # Người có thu nhập cao có khả năng mua bảo hiểm đầu tư cao hơn
    if row['income_monthly'] > 80_000_000:
        probs[3] += 0.2 # Index 3 là 'Bảo hiểm Liên kết Đầu tư'
    # Người đã kết hôn và có con có khả năng mua BH cho gia đình
    if row['marital_status'] == 'Đã kết hôn' and row['num_children'] > 0:
        probs[6] += 0.2 # Index 6 là 'BH Sức khỏe cho Gia đình'
        
    # Chuẩn hóa lại xác suất để tổng bằng 1
    probs = np.array(probs) / sum(probs)
    
    # Chọn một sản phẩm dựa trên xác suất đã điều chỉnh
    purchased_products.append(np.random.choice(product_ids, p=probs))

df['purchased_product_id'] = purchased_products

# --- Tạo giá trị thiếu (missing values) ---
# df.loc[df.sample(frac=0.05).index, 'age'] = np.nan
# df.loc[df.sample(frac=0.10).index, 'income_monthly'] = np.nan
# df.loc[df.sample(frac=0.02).index, 'marital_status'] = np.nan

# --- Lưu vào file CSV ---
output_filename = 'raw_customer_and_purchase_data.csv'
# Sắp xếp lại cột để dễ nhìn hơn
df = df[[
    'user_id', 'full_name', 'age', 'gender', 'marital_status', 'num_children',
    'location', 'job_title', 'income_monthly', 'purchased_product_id'
]]
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\nĐã tạo và lưu dữ liệu thành công vào file: {output_filename}")
print("\n5 dòng dữ liệu đầu tiên:")
print(df.head())

print("\nThống kê số lượng các gói bảo hiểm đã được mua:")
print(df['purchased_product_id'].value_counts())


import pandas as pd
import os

print("Bắt đầu tạo file mô tả các gói bảo hiểm...")

# Định nghĩa thông tin chi tiết cho từng sản phẩm
# Đây là "bộ não" chứa kiến thức về sản phẩm của chúng ta
products_data = {
    'P01': {
        'name': 'Bảo hiểm Sức khỏe Toàn diện',
        'description': """
Đối tượng phù hợp: Mọi cá nhân và gia đình mong muốn được bảo vệ tài chính trước các rủi ro về sức khỏe, từ khám chữa bệnh thông thường đến điều trị nội trú phức tạp.
Quyền lợi chính:
- Chi trả 100% chi phí điều trị nội trú, phẫu thuật, và chi phí phòng, giường bệnh.
- Quyền lợi điều trị ngoại trú, nha khoa, và thai sản tùy chọn.
- Bảo lãnh viện phí tại hàng trăm bệnh viện và phòng khám chất lượng cao trên toàn quốc.
Điểm nổi bật: Thủ tục bồi thường nhanh gọn, không giới hạn số lần khám chữa bệnh. Là tấm lá chắn tài chính vững chắc cho sức khỏe của bạn.
"""
    },
    'P02': {
        'name': 'Bảo hiểm Bệnh hiểm nghèo',
        'description': """
Đối tượng phù hợp: Người trưởng thành, đặc biệt là trụ cột kinh tế trong gia đình, muốn có một quỹ dự phòng lớn để đối phó với các bệnh lý nghiêm trọng.
Quyền lợi chính:
- Chi trả một lần toàn bộ số tiền bảo hiểm ngay khi có chẩn đoán mắc một trong các bệnh hiểm nghèo theo danh mục (ung thư, đột quỵ, suy thận, ...).
- Hỗ trợ tài chính kịp thời để trang trải chi phí điều trị đắt đỏ và bù đắp thu nhập bị mất.
- Quyền lợi có thể được chi trả ở nhiều giai đoạn bệnh khác nhau.
Điểm nổi bật: Phí bảo hiểm hợp lý, quyền lợi chi trả lớn, giúp bạn an tâm chiến đấu với bệnh tật mà không phải lo lắng về gánh nặng tài chính.
"""
    },
    'P03': {
        'name': 'Bảo hiểm Nhân thọ Tích lũy',
        'description': """
Đối tượng phù hợp: Những người có kế hoạch tài chính dài hạn, vừa muốn được bảo vệ trước rủi ro tử vong hoặc thương tật, vừa muốn xây dựng một quỹ tiết kiệm có kỷ luật.
Quyền lợi chính:
- Bảo vệ tài chính cho gia đình trước rủi ro tử vong hoặc thương tật toàn bộ vĩnh viễn của người được bảo hiểm.
- Nhận lại toàn bộ giá trị tài khoản hợp đồng khi đáo hạn, bao gồm gốc và lãi tích lũy.
- Các khoản thưởng duy trì hợp đồng định kỳ hấp dẫn.
Điểm nổi bật: Giải pháp 2 trong 1: Bảo vệ vững chắc và Tích lũy an toàn. Xây dựng tương lai bền vững cho bản thân và những người thân yêu.
"""
    },
    'P04': {
        'name': 'Bảo hiểm Liên kết Đầu tư',
        'description': """
Đối tượng phù hợp: Khách hàng có khẩu vị rủi ro, am hiểu về đầu tư và mong muốn gia tăng tài sản một cách linh hoạt thông qua các quỹ đầu tư chuyên nghiệp.
Quyền lợi chính:
- Quyền lợi bảo vệ nhân thọ trước các rủi ro không lường trước.
- Linh hoạt lựa chọn các quỹ đầu tư (cổ phiếu, trái phiếu) để tối ưu hóa lợi nhuận kỳ vọng.
- Dễ dàng thay đổi tỷ lệ phân bổ đầu tư, rút tiền, hoặc đóng thêm phí để gia tăng giá trị tài khoản.
Điểm nổi bật: Tối đa hóa tiềm năng tăng trưởng tài sản trong dài hạn, đồng thời vẫn duy trì một lớp bảo vệ tài chính cốt lõi.
"""
    },
    'P05': {
        'name': 'Bảo hiểm Hưu trí An nhàn',
        'description': """
Đối tượng phù hợp: Người lao động đang trong độ tuổi tích lũy, mong muốn có một nguồn thu nhập ổn định và độc lập về tài chính khi về hưu.
Quyền lợi chính:
- Tích lũy tài sản một cách có hệ thống trong suốt quá trình làm việc.
- Nhận quyền lợi hưu trí định kỳ (hàng tháng, hàng quý) sau khi đến tuổi nghỉ hưu.
- Vẫn được bảo vệ trước rủi ro tử vong hoặc thương tật trong thời gian đóng phí.
Điểm nổi bật: Đảm bảo một tuổi già an nhàn, độc lập, không phụ thuộc vào con cháu. Bắt đầu kế hoạch hưu trí của bạn ngay hôm nay.
"""
    },
    'P06': {
        'name': 'Bảo hiểm Tai nạn Cá nhân',
        'description': """
Đối tượng phù hợp: Mọi người, đặc biệt là những người thường xuyên di chuyển, làm việc trong môi trường có rủi ro cao hoặc tham gia các hoạt động thể thao.
Quyền lợi chính:
- Chi trả chi phí y tế phát sinh do tai nạn.
- Trợ cấp thu nhập trong thời gian nằm viện điều trị thương tật do tai nạn.
- Chi trả số tiền bảo hiểm lớn trong trường hợp tử vong hoặc thương tật toàn bộ vĩnh viễn do tai nạn.
Điểm nổi bật: Phạm vi bảo vệ 24/7 trên toàn thế giới. Mức phí cực kỳ thấp nhưng mang lại sự bảo vệ thiết thực trước những rủi ro bất ngờ nhất trong cuộc sống.
"""
    },
    'P07': {
        'name': 'Bảo hiểm Sức khỏe cho Gia đình',
        'description': """
Đối tượng phù hợp: Các gia đình có con nhỏ, muốn bảo vệ sức khỏe cho tất cả thành viên chỉ trong một hợp đồng duy nhất.
Quyền lợi chính:
- Tất cả thành viên trong gia đình (vợ, chồng, con cái) được bảo vệ chung trên một hợp đồng.
- Hạn mức bảo hiểm chung cho cả gia đình hoặc riêng cho từng thành viên.
- Bao gồm đầy đủ các quyền lợi nội trú, ngoại trú, nha khoa.
Điểm nổi bật: Tiết kiệm chi phí và quản lý thuận tiện hơn so với việc mua nhiều hợp đồng riêng lẻ. Sự lựa chọn thông minh để bảo vệ tổ ấm của bạn.
"""
    }
}

# Chuyển đổi từ dictionary sang danh sách các dòng dữ liệu
product_list = []
for pid, details in products_data.items():
    # Loại bỏ các khoảng trắng thừa và xuống dòng ở đầu/cuối mô tả
    cleaned_description = details['description'].strip()
    product_list.append({
        'product_id': pid,
        'product_name': details['name'],
        'description': cleaned_description
    })

# Tạo DataFrame từ danh sách
df_products = pd.DataFrame(product_list)

# --- Lưu vào file CSV ---
output_filename = 'Train_insurance_products_description.csv'
df_products.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\nĐã tạo và lưu dữ liệu mô tả sản phẩm thành công vào file: {output_filename}")
print("\nNội dung file:")
print(df_products)
