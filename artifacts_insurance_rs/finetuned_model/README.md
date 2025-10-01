---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:20000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 'User Profile:

    - Mã khách hàng: 4237

    - Tuổi: 62

    - Giới tính: Nam

    - Tình trạng hôn nhân: Đã kết hôn

    - Số con: 2

    - Nơi ở: TP. HCM

    - Nghề nghiệp: Kỹ sư phần mềm

    - Thu nhập/tháng: 142000000

    Goal: find the most suitable insurance product for this profile.'
  sentences:
  - 'Insurance Product:

    - Mã sản phẩm: P01

    - Tên sản phẩm: Bảo hiểm Sức khỏe Toàn diện

    - Mô tả: Đối tượng phù hợp: Mọi cá nhân và gia đình mong muốn được bảo vệ tài
    chính trước các rủi ro về sức khỏe, từ khám chữa bệnh thông thường đến điều trị
    nội trú phức tạp.

    Quyền lợi chính:

    - Chi trả 100% chi phí điều trị nội trú, phẫu thuật, và chi phí phòng, giường
    bệnh.

    - Quyền lợi điều trị ngoại trú, nha khoa, và thai sản tùy chọn.

    - Bảo lãnh viện phí tại hàng trăm bệnh viện và phòng khám chất lượng cao trên
    toàn quốc.

    Điểm nổi bật: Thủ tục bồi thường nhanh gọn, không giới hạn số lần khám chữa bệnh.
    Là tấm lá chắn tài chính vững chắc cho sức khỏe của bạn.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P04

    - Tên sản phẩm: Bảo hiểm Liên kết Đầu tư

    - Mô tả: Đối tượng phù hợp: Khách hàng có khẩu vị rủi ro, am hiểu về đầu tư và
    mong muốn gia tăng tài sản một cách linh hoạt thông qua các quỹ đầu tư chuyên
    nghiệp.

    Quyền lợi chính:

    - Quyền lợi bảo vệ nhân thọ trước các rủi ro không lường trước.

    - Linh hoạt lựa chọn các quỹ đầu tư (cổ phiếu, trái phiếu) để tối ưu hóa lợi nhuận
    kỳ vọng.

    - Dễ dàng thay đổi tỷ lệ phân bổ đầu tư, rút tiền, hoặc đóng thêm phí để gia tăng
    giá trị tài khoản.

    Điểm nổi bật: Tối đa hóa tiềm năng tăng trưởng tài sản trong dài hạn, đồng thời
    vẫn duy trì một lớp bảo vệ tài chính cốt lõi.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P07

    - Tên sản phẩm: Bảo hiểm Sức khỏe cho Gia đình

    - Mô tả: Đối tượng phù hợp: Các gia đình có con nhỏ, muốn bảo vệ sức khỏe cho
    tất cả thành viên chỉ trong một hợp đồng duy nhất.

    Quyền lợi chính:

    - Tất cả thành viên trong gia đình (vợ, chồng, con cái) được bảo vệ chung trên
    một hợp đồng.

    - Hạn mức bảo hiểm chung cho cả gia đình hoặc riêng cho từng thành viên.

    - Bao gồm đầy đủ các quyền lợi nội trú, ngoại trú, nha khoa.

    Điểm nổi bật: Tiết kiệm chi phí và quản lý thuận tiện hơn so với việc mua nhiều
    hợp đồng riêng lẻ. Sự lựa chọn thông minh để bảo vệ tổ ấm của bạn.

    Goal: match to users who would benefit most.'
- source_sentence: 'User Profile:

    - Mã khách hàng: 9027

    - Tuổi: 32

    - Giới tính: Nữ

    - Tình trạng hôn nhân: Độc thân

    - Số con: 0

    - Nơi ở: Hà Nội

    - Nghề nghiệp: Giáo viên

    - Thu nhập/tháng: 36000000

    Goal: find the most suitable insurance product for this profile.'
  sentences:
  - 'Insurance Product:

    - Mã sản phẩm: P06

    - Tên sản phẩm: Bảo hiểm Tai nạn Cá nhân

    - Mô tả: Đối tượng phù hợp: Mọi người, đặc biệt là những người thường xuyên di
    chuyển, làm việc trong môi trường có rủi ro cao hoặc tham gia các hoạt động thể
    thao.

    Quyền lợi chính:

    - Chi trả chi phí y tế phát sinh do tai nạn.

    - Trợ cấp thu nhập trong thời gian nằm viện điều trị thương tật do tai nạn.

    - Chi trả số tiền bảo hiểm lớn trong trường hợp tử vong hoặc thương tật toàn bộ
    vĩnh viễn do tai nạn.

    Điểm nổi bật: Phạm vi bảo vệ 24/7 trên toàn thế giới. Mức phí cực kỳ thấp nhưng
    mang lại sự bảo vệ thiết thực trước những rủi ro bất ngờ nhất trong cuộc sống.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P01

    - Tên sản phẩm: Bảo hiểm Sức khỏe Toàn diện

    - Mô tả: Đối tượng phù hợp: Mọi cá nhân và gia đình mong muốn được bảo vệ tài
    chính trước các rủi ro về sức khỏe, từ khám chữa bệnh thông thường đến điều trị
    nội trú phức tạp.

    Quyền lợi chính:

    - Chi trả 100% chi phí điều trị nội trú, phẫu thuật, và chi phí phòng, giường
    bệnh.

    - Quyền lợi điều trị ngoại trú, nha khoa, và thai sản tùy chọn.

    - Bảo lãnh viện phí tại hàng trăm bệnh viện và phòng khám chất lượng cao trên
    toàn quốc.

    Điểm nổi bật: Thủ tục bồi thường nhanh gọn, không giới hạn số lần khám chữa bệnh.
    Là tấm lá chắn tài chính vững chắc cho sức khỏe của bạn.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P03

    - Tên sản phẩm: Bảo hiểm Nhân thọ Tích lũy

    - Mô tả: Đối tượng phù hợp: Những người có kế hoạch tài chính dài hạn, vừa muốn
    được bảo vệ trước rủi ro tử vong hoặc thương tật, vừa muốn xây dựng một quỹ tiết
    kiệm có kỷ luật.

    Quyền lợi chính:

    - Bảo vệ tài chính cho gia đình trước rủi ro tử vong hoặc thương tật toàn bộ vĩnh
    viễn của người được bảo hiểm.

    - Nhận lại toàn bộ giá trị tài khoản hợp đồng khi đáo hạn, bao gồm gốc và lãi
    tích lũy.

    - Các khoản thưởng duy trì hợp đồng định kỳ hấp dẫn.

    Điểm nổi bật: Giải pháp 2 trong 1: Bảo vệ vững chắc và Tích lũy an toàn. Xây dựng
    tương lai bền vững cho bản thân và những người thân yêu.

    Goal: match to users who would benefit most.'
- source_sentence: 'User Profile:

    - Mã khách hàng: 16126

    - Tuổi: 20

    - Giới tính: Nam

    - Tình trạng hôn nhân: Độc thân

    - Số con: 0

    - Nơi ở: Hải Phòng

    - Nghề nghiệp: Kỹ sư phần mềm

    - Thu nhập/tháng: 39000000

    Goal: find the most suitable insurance product for this profile.'
  sentences:
  - 'Insurance Product:

    - Mã sản phẩm: P04

    - Tên sản phẩm: Bảo hiểm Liên kết Đầu tư

    - Mô tả: Đối tượng phù hợp: Khách hàng có khẩu vị rủi ro, am hiểu về đầu tư và
    mong muốn gia tăng tài sản một cách linh hoạt thông qua các quỹ đầu tư chuyên
    nghiệp.

    Quyền lợi chính:

    - Quyền lợi bảo vệ nhân thọ trước các rủi ro không lường trước.

    - Linh hoạt lựa chọn các quỹ đầu tư (cổ phiếu, trái phiếu) để tối ưu hóa lợi nhuận
    kỳ vọng.

    - Dễ dàng thay đổi tỷ lệ phân bổ đầu tư, rút tiền, hoặc đóng thêm phí để gia tăng
    giá trị tài khoản.

    Điểm nổi bật: Tối đa hóa tiềm năng tăng trưởng tài sản trong dài hạn, đồng thời
    vẫn duy trì một lớp bảo vệ tài chính cốt lõi.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P07

    - Tên sản phẩm: Bảo hiểm Sức khỏe cho Gia đình

    - Mô tả: Đối tượng phù hợp: Các gia đình có con nhỏ, muốn bảo vệ sức khỏe cho
    tất cả thành viên chỉ trong một hợp đồng duy nhất.

    Quyền lợi chính:

    - Tất cả thành viên trong gia đình (vợ, chồng, con cái) được bảo vệ chung trên
    một hợp đồng.

    - Hạn mức bảo hiểm chung cho cả gia đình hoặc riêng cho từng thành viên.

    - Bao gồm đầy đủ các quyền lợi nội trú, ngoại trú, nha khoa.

    Điểm nổi bật: Tiết kiệm chi phí và quản lý thuận tiện hơn so với việc mua nhiều
    hợp đồng riêng lẻ. Sự lựa chọn thông minh để bảo vệ tổ ấm của bạn.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P04

    - Tên sản phẩm: Bảo hiểm Liên kết Đầu tư

    - Mô tả: Đối tượng phù hợp: Khách hàng có khẩu vị rủi ro, am hiểu về đầu tư và
    mong muốn gia tăng tài sản một cách linh hoạt thông qua các quỹ đầu tư chuyên
    nghiệp.

    Quyền lợi chính:

    - Quyền lợi bảo vệ nhân thọ trước các rủi ro không lường trước.

    - Linh hoạt lựa chọn các quỹ đầu tư (cổ phiếu, trái phiếu) để tối ưu hóa lợi nhuận
    kỳ vọng.

    - Dễ dàng thay đổi tỷ lệ phân bổ đầu tư, rút tiền, hoặc đóng thêm phí để gia tăng
    giá trị tài khoản.

    Điểm nổi bật: Tối đa hóa tiềm năng tăng trưởng tài sản trong dài hạn, đồng thời
    vẫn duy trì một lớp bảo vệ tài chính cốt lõi.

    Goal: match to users who would benefit most.'
- source_sentence: 'User Profile:

    - Mã khách hàng: 17390

    - Tuổi: 59

    - Giới tính: Nam

    - Tình trạng hôn nhân: Đã kết hôn

    - Số con: 2

    - Nơi ở: Hà Nội

    - Nghề nghiệp: Kỹ sư phần mềm

    - Thu nhập/tháng: 49000000

    Goal: find the most suitable insurance product for this profile.'
  sentences:
  - 'Insurance Product:

    - Mã sản phẩm: P03

    - Tên sản phẩm: Bảo hiểm Nhân thọ Tích lũy

    - Mô tả: Đối tượng phù hợp: Những người có kế hoạch tài chính dài hạn, vừa muốn
    được bảo vệ trước rủi ro tử vong hoặc thương tật, vừa muốn xây dựng một quỹ tiết
    kiệm có kỷ luật.

    Quyền lợi chính:

    - Bảo vệ tài chính cho gia đình trước rủi ro tử vong hoặc thương tật toàn bộ vĩnh
    viễn của người được bảo hiểm.

    - Nhận lại toàn bộ giá trị tài khoản hợp đồng khi đáo hạn, bao gồm gốc và lãi
    tích lũy.

    - Các khoản thưởng duy trì hợp đồng định kỳ hấp dẫn.

    Điểm nổi bật: Giải pháp 2 trong 1: Bảo vệ vững chắc và Tích lũy an toàn. Xây dựng
    tương lai bền vững cho bản thân và những người thân yêu.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P05

    - Tên sản phẩm: Bảo hiểm Hưu trí An nhàn

    - Mô tả: Đối tượng phù hợp: Người lao động đang trong độ tuổi tích lũy, mong muốn
    có một nguồn thu nhập ổn định và độc lập về tài chính khi về hưu.

    Quyền lợi chính:

    - Tích lũy tài sản một cách có hệ thống trong suốt quá trình làm việc.

    - Nhận quyền lợi hưu trí định kỳ (hàng tháng, hàng quý) sau khi đến tuổi nghỉ
    hưu.

    - Vẫn được bảo vệ trước rủi ro tử vong hoặc thương tật trong thời gian đóng phí.

    Điểm nổi bật: Đảm bảo một tuổi già an nhàn, độc lập, không phụ thuộc vào con cháu.
    Bắt đầu kế hoạch hưu trí của bạn ngay hôm nay.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P01

    - Tên sản phẩm: Bảo hiểm Sức khỏe Toàn diện

    - Mô tả: Đối tượng phù hợp: Mọi cá nhân và gia đình mong muốn được bảo vệ tài
    chính trước các rủi ro về sức khỏe, từ khám chữa bệnh thông thường đến điều trị
    nội trú phức tạp.

    Quyền lợi chính:

    - Chi trả 100% chi phí điều trị nội trú, phẫu thuật, và chi phí phòng, giường
    bệnh.

    - Quyền lợi điều trị ngoại trú, nha khoa, và thai sản tùy chọn.

    - Bảo lãnh viện phí tại hàng trăm bệnh viện và phòng khám chất lượng cao trên
    toàn quốc.

    Điểm nổi bật: Thủ tục bồi thường nhanh gọn, không giới hạn số lần khám chữa bệnh.
    Là tấm lá chắn tài chính vững chắc cho sức khỏe của bạn.

    Goal: match to users who would benefit most.'
- source_sentence: 'User Profile:

    - Mã khách hàng: 8247

    - Tuổi: 36

    - Giới tính: Nữ

    - Tình trạng hôn nhân: Đã kết hôn

    - Số con: 1

    - Nơi ở: Đà Nẵng

    - Nghề nghiệp: Giáo viên

    - Thu nhập/tháng: 99000000

    Goal: find the most suitable insurance product for this profile.'
  sentences:
  - 'Insurance Product:

    - Mã sản phẩm: P04

    - Tên sản phẩm: Bảo hiểm Liên kết Đầu tư

    - Mô tả: Đối tượng phù hợp: Khách hàng có khẩu vị rủi ro, am hiểu về đầu tư và
    mong muốn gia tăng tài sản một cách linh hoạt thông qua các quỹ đầu tư chuyên
    nghiệp.

    Quyền lợi chính:

    - Quyền lợi bảo vệ nhân thọ trước các rủi ro không lường trước.

    - Linh hoạt lựa chọn các quỹ đầu tư (cổ phiếu, trái phiếu) để tối ưu hóa lợi nhuận
    kỳ vọng.

    - Dễ dàng thay đổi tỷ lệ phân bổ đầu tư, rút tiền, hoặc đóng thêm phí để gia tăng
    giá trị tài khoản.

    Điểm nổi bật: Tối đa hóa tiềm năng tăng trưởng tài sản trong dài hạn, đồng thời
    vẫn duy trì một lớp bảo vệ tài chính cốt lõi.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P01

    - Tên sản phẩm: Bảo hiểm Sức khỏe Toàn diện

    - Mô tả: Đối tượng phù hợp: Mọi cá nhân và gia đình mong muốn được bảo vệ tài
    chính trước các rủi ro về sức khỏe, từ khám chữa bệnh thông thường đến điều trị
    nội trú phức tạp.

    Quyền lợi chính:

    - Chi trả 100% chi phí điều trị nội trú, phẫu thuật, và chi phí phòng, giường
    bệnh.

    - Quyền lợi điều trị ngoại trú, nha khoa, và thai sản tùy chọn.

    - Bảo lãnh viện phí tại hàng trăm bệnh viện và phòng khám chất lượng cao trên
    toàn quốc.

    Điểm nổi bật: Thủ tục bồi thường nhanh gọn, không giới hạn số lần khám chữa bệnh.
    Là tấm lá chắn tài chính vững chắc cho sức khỏe của bạn.

    Goal: match to users who would benefit most.'
  - 'Insurance Product:

    - Mã sản phẩm: P04

    - Tên sản phẩm: Bảo hiểm Liên kết Đầu tư

    - Mô tả: Đối tượng phù hợp: Khách hàng có khẩu vị rủi ro, am hiểu về đầu tư và
    mong muốn gia tăng tài sản một cách linh hoạt thông qua các quỹ đầu tư chuyên
    nghiệp.

    Quyền lợi chính:

    - Quyền lợi bảo vệ nhân thọ trước các rủi ro không lường trước.

    - Linh hoạt lựa chọn các quỹ đầu tư (cổ phiếu, trái phiếu) để tối ưu hóa lợi nhuận
    kỳ vọng.

    - Dễ dàng thay đổi tỷ lệ phân bổ đầu tư, rút tiền, hoặc đóng thêm phí để gia tăng
    giá trị tài khoản.

    Điểm nổi bật: Tối đa hóa tiềm năng tăng trưởng tài sản trong dài hạn, đồng thời
    vẫn duy trì một lớp bảo vệ tài chính cốt lõi.

    Goal: match to users who would benefit most.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'User Profile:\n- Mã khách hàng: 8247\n- Tuổi: 36\n- Giới tính: Nữ\n- Tình trạng hôn nhân: Đã kết hôn\n- Số con: 1\n- Nơi ở: Đà Nẵng\n- Nghề nghiệp: Giáo viên\n- Thu nhập/tháng: 99000000\nGoal: find the most suitable insurance product for this profile.',
    'Insurance Product:\n- Mã sản phẩm: P04\n- Tên sản phẩm: Bảo hiểm Liên kết Đầu tư\n- Mô tả: Đối tượng phù hợp: Khách hàng có khẩu vị rủi ro, am hiểu về đầu tư và mong muốn gia tăng tài sản một cách linh hoạt thông qua các quỹ đầu tư chuyên nghiệp.\nQuyền lợi chính:\n- Quyền lợi bảo vệ nhân thọ trước các rủi ro không lường trước.\n- Linh hoạt lựa chọn các quỹ đầu tư (cổ phiếu, trái phiếu) để tối ưu hóa lợi nhuận kỳ vọng.\n- Dễ dàng thay đổi tỷ lệ phân bổ đầu tư, rút tiền, hoặc đóng thêm phí để gia tăng giá trị tài khoản.\nĐiểm nổi bật: Tối đa hóa tiềm năng tăng trưởng tài sản trong dài hạn, đồng thời vẫn duy trì một lớp bảo vệ tài chính cốt lõi.\nGoal: match to users who would benefit most.',
    'Insurance Product:\n- Mã sản phẩm: P01\n- Tên sản phẩm: Bảo hiểm Sức khỏe Toàn diện\n- Mô tả: Đối tượng phù hợp: Mọi cá nhân và gia đình mong muốn được bảo vệ tài chính trước các rủi ro về sức khỏe, từ khám chữa bệnh thông thường đến điều trị nội trú phức tạp.\nQuyền lợi chính:\n- Chi trả 100% chi phí điều trị nội trú, phẫu thuật, và chi phí phòng, giường bệnh.\n- Quyền lợi điều trị ngoại trú, nha khoa, và thai sản tùy chọn.\n- Bảo lãnh viện phí tại hàng trăm bệnh viện và phòng khám chất lượng cao trên toàn quốc.\nĐiểm nổi bật: Thủ tục bồi thường nhanh gọn, không giới hạn số lần khám chữa bệnh. Là tấm lá chắn tài chính vững chắc cho sức khỏe của bạn.\nGoal: match to users who would benefit most.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8219, 0.8064],
#         [0.8219, 1.0000, 0.9600],
#         [0.8064, 0.9600, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 20,000 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                            |
  |:--------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                                |
  | details | <ul><li>min: 84 tokens</li><li>mean: 89.81 tokens</li><li>max: 97 tokens</li></ul> | <ul><li>min: 247 tokens</li><li>mean: 255.12 tokens</li><li>max: 256 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                         | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>User Profile:<br>- Mã khách hàng: 4210<br>- Tuổi: 47<br>- Giới tính: Nam<br>- Tình trạng hôn nhân: Đã kết hôn<br>- Số con: 1<br>- Nơi ở: Hà Nội<br>- Nghề nghiệp: Sinh viên<br>- Thu nhập/tháng: 86000000<br>Goal: find the most suitable insurance product for this profile.</code>         | <code>Insurance Product:<br>- Mã sản phẩm: P04<br>- Tên sản phẩm: Bảo hiểm Liên kết Đầu tư<br>- Mô tả: Đối tượng phù hợp: Khách hàng có khẩu vị rủi ro, am hiểu về đầu tư và mong muốn gia tăng tài sản một cách linh hoạt thông qua các quỹ đầu tư chuyên nghiệp.<br>Quyền lợi chính:<br>- Quyền lợi bảo vệ nhân thọ trước các rủi ro không lường trước.<br>- Linh hoạt lựa chọn các quỹ đầu tư (cổ phiếu, trái phiếu) để tối ưu hóa lợi nhuận kỳ vọng.<br>- Dễ dàng thay đổi tỷ lệ phân bổ đầu tư, rút tiền, hoặc đóng thêm phí để gia tăng giá trị tài khoản.<br>Điểm nổi bật: Tối đa hóa tiềm năng tăng trưởng tài sản trong dài hạn, đồng thời vẫn duy trì một lớp bảo vệ tài chính cốt lõi.<br>Goal: match to users who would benefit most.</code>                                                                 |
  | <code>User Profile:<br>- Mã khách hàng: 13586<br>- Tuổi: 45<br>- Giới tính: Nam<br>- Tình trạng hôn nhân: Đã kết hôn<br>- Số con: 1<br>- Nơi ở: Hải Phòng<br>- Nghề nghiệp: Lao động tự do<br>- Thu nhập/tháng: 7000000<br>Goal: find the most suitable insurance product for this profile.</code> | <code>Insurance Product:<br>- Mã sản phẩm: P07<br>- Tên sản phẩm: Bảo hiểm Sức khỏe cho Gia đình<br>- Mô tả: Đối tượng phù hợp: Các gia đình có con nhỏ, muốn bảo vệ sức khỏe cho tất cả thành viên chỉ trong một hợp đồng duy nhất.<br>Quyền lợi chính:<br>- Tất cả thành viên trong gia đình (vợ, chồng, con cái) được bảo vệ chung trên một hợp đồng.<br>- Hạn mức bảo hiểm chung cho cả gia đình hoặc riêng cho từng thành viên.<br>- Bao gồm đầy đủ các quyền lợi nội trú, ngoại trú, nha khoa.<br>Điểm nổi bật: Tiết kiệm chi phí và quản lý thuận tiện hơn so với việc mua nhiều hợp đồng riêng lẻ. Sự lựa chọn thông minh để bảo vệ tổ ấm của bạn.<br>Goal: match to users who would benefit most.</code>                                                                                                        |
  | <code>User Profile:<br>- Mã khách hàng: 10010<br>- Tuổi: 29<br>- Giới tính: Nữ<br>- Tình trạng hôn nhân: Đã kết hôn<br>- Số con: 2<br>- Nơi ở: Bình Dương<br>- Nghề nghiệp: Công nhân<br>- Thu nhập/tháng: 93000000<br>Goal: find the most suitable insurance product for this profile.</code>     | <code>Insurance Product:<br>- Mã sản phẩm: P02<br>- Tên sản phẩm: Bảo hiểm Bệnh hiểm nghèo<br>- Mô tả: Đối tượng phù hợp: Người trưởng thành, đặc biệt là trụ cột kinh tế trong gia đình, muốn có một quỹ dự phòng lớn để đối phó với các bệnh lý nghiêm trọng.<br>Quyền lợi chính:<br>- Chi trả một lần toàn bộ số tiền bảo hiểm ngay khi có chẩn đoán mắc một trong các bệnh hiểm nghèo theo danh mục (ung thư, đột quỵ, suy thận, ...).<br>- Hỗ trợ tài chính kịp thời để trang trải chi phí điều trị đắt đỏ và bù đắp thu nhập bị mất.<br>- Quyền lợi có thể được chi trả ở nhiều giai đoạn bệnh khác nhau.<br>Điểm nổi bật: Phí bảo hiểm hợp lý, quyền lợi chi trả lớn, giúp bạn an tâm chiến đấu với bệnh tật mà không phải lo lắng về gánh nặng tài chính.<br>Goal: match to users who would benefit most.</code> |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
- `num_train_epochs`: 10
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 128
- `per_device_eval_batch_size`: 128
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 10
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 3.1847 | 500  | 4.8131        |
| 6.3694 | 1000 | 4.7814        |
| 9.5541 | 1500 | 4.772         |


### Framework Versions
- Python: 3.10.18
- Sentence Transformers: 5.1.1
- Transformers: 4.56.2
- PyTorch: 2.8.0+cu129
- Accelerate: 1.10.1
- Datasets: 4.1.1
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->