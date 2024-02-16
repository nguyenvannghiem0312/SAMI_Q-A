# SAMI Question and Answering

This is a repository containing the source code of project 2, implementing a chatbot using the RAG technique - Retrieval Augmented Generation to inquire about some basic information about the School of Applied Mathematics and Informatics - Hanoi University of Science and Technology (Now renamed as the Faculty of Mathematics and Informatics).

![alt text](https://github.com/nguyenvannghiem0312/SAMI_Q-A/blob/main/Figure/model.png)

To use the model, you need to download the following two models:
- SBert Model: [SBert Model](https://huggingface.co/NghiemAbe/sami-sbert-CT)
- VinaLLaMa-7B-Chat Model: [VinaLLaMa-7B-Chat Model](https://huggingface.co/vilm/vinallama-7b-chat-GGUF)

You can replace these models as needed.

Using the SBert model, I fine-tune the pre-trained model at [Vietnamese Bi-Encoder Model](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder), and utilized the ContrastiveTensionInBatchNegativeLoss function. I compared the results with the original model at [PhoBERT](https://github.com/VinAIResearch/PhoBERT):

![alt text](https://github.com/nguyenvannghiem0312/SAMI_Q-A/blob/main/Figure/MRRandMAP.png)

Combining with BM25 in a ratio of 3:7, I achieved impressive results:

Weight = 0.3 * BM25 + 0.7 Embedding

| MRR@k | Value | Recall@k | Value |
|-------|-------|----------|-------|
| MRR@1 | 0.7988 | Recall@1 | 0.7988 |
| MRR@2 | 0.8526 | Recall@2 | 0.9065 |
| MRR@3 | 0.8662 | Recall@3 | 0.9472 |
| MRR@4 | 0.8703 | Recall@4 | 0.9634 |
| MRR@5 | 0.8719 | Recall@5 | 0.9715 |
| MRR@6 | 0.8729 | Recall@6 | 0.9776 |
| MRR@7 | 0.8732 | Recall@7 | 0.9797 |
| MRR@8 | 0.8734 | Recall@8 | 0.9817 |
| MRR@9 | 0.8739 | Recall@9 | 0.9858 |
| MRR@10 | 0.8739 | Recall@10 | 0.9858 |

Here are some examples from the model:

**User:** Học Toán tin sau ra làm gì?
**System:** Sinh viên chuyên ngành Toán-Tin có thể công tác trong nhiều ngành nghề sau khi tốt nghiệp, bao gồm:
1. Nhà phân tích hệ thống
2. Kỹ sư phần mềm
3. Chuyên gia tin học
4. Chuyên viên thống kê
5. Nhà toán học
6. Giảng viên đại học
Time: 23.51

**User:** Sinh viên viện Toán học xong làm được gì?
**System:** Sau khi tốt nghiệp từ Viện Toán ứng dụng và Tin học, sinh viên có thể tìm được việc làm trong các lĩnh vực như ngân hàng, trung tâm phân tích tài chính, cơ quan kiểm toán thống kê, Tổng cục Thống kê, Ủy ban kế hoạch Nhà nước, trường đại học, cơ sở nghiên cứu, hoặc trong ngành tin học. Họ cũng có thể tiếp tục học sau đại học tại nước ngoài. Đa số cựu sinh viên của Viện có thu nhập khá cao và ổn định, và một số người đã trở thành các thành viên chủ chốt của đơn vị công tác.
Time: 114.39
