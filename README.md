==========Multimodal Fake News Detection (BERT + ResNet50)==========

Dự án này sử dụng phương pháp học sâu đa phương thức (Multimodal Deep Learning) để phân loại tin tức giả (Fake News) dựa trên cả nội dung văn bản và hình ảnh đi kèm. Mô hình kết hợp sức mạnh của BERT (xử lý ngôn ngữ tự nhiên) và ResNet50 (xử lý hình ảnh) để tối ưu hóa độ chính xác trong việc nhận diện thông tin sai lệch.


  📖 Giới thiệu
  
Tin giả ngày nay không chỉ nằm ở câu chữ mà còn được lồng ghép qua hình ảnh đánh lừa thị giác. Dự án này giải quyết vấn đề đó bằng cách:

  -Trích xuất đặc trưng văn bản: Sử dụng bert-base-uncased.
  
  -Trích xuất đặc trưng hình ảnh: Sử dụng ResNet50 (Pre-trained).
  
  -Late Fusion: Kết hợp vector đặc trưng của cả hai thông qua một lớp Fusion (Linear Layer) để đưa ra quyết định cuối cùng.

  
  🏗 Kiến trúc mô hìnhMô hình được xây dựng bằng PyTorch với các thành phần chính:
  
    -Text Branch: BERT Model lấy đầu ra là pooler_output (768 chiều).        
    
    -Image Branch: ResNet50 loại bỏ lớp phân loại cuối (Identity), giữ lại vector đặc trưng (2048 chiều).        
    
    -Fusion Layer: Kết hợp 2 vector (768+2048=2816) đi qua lớp Fully Connected với Dropout (0.7) để tránh overfitting.

    
  📊 Dữ liệu sử dụng
  
Dự án sử dụng tập dữ liệu Fakeddit, một tập dữ liệu đa phương thức quy mô lớn dành cho nghiên cứu tin giả.

Nguồn: Kaggle - Fakeddit Dataset

Số lượng sử dụng: 16,000 mẫu đã được cân bằng (8,000 Real / 8,000 Fake).

Tiền xử lý:

-Văn bản: Tokenize bởi BERT Tokenizer, max length 64.

-Hình ảnh: Resize (224x224), chuẩn hóa theo ImageNet.


🔗 Liên kết tham khảo

Hugging Face: https://huggingface.co/spaces/kiet15122005/FakeNews-Multimodal-Detection

Dataset: https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset


----------------------------------------------------------------------------------
Dự án được thực hiện nhằm mục đích học tập và nghiên cứu về Multimodal AI.
----------------------------------------------------------------------------------
