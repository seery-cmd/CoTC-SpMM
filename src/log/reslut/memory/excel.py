import openpyxl

def txt_to_excel(txt_path, excel_path, sheet_name="Sheet1"):
    """
    读取 txt 文件，按空格分割每行，将分割后的元素写入 Excel 单元格。

    参数:
        txt_path (str): 输入的 txt 文件路径
        excel_path (str): 输出的 Excel 文件路径（支持 .xlsx）
        sheet_name (str): 工作表名称，默认为 "Sheet1"
    """
    # 创建一个新的 Excel 工作簿
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    row = 2
    with open(txt_path, 'r', encoding='utf-8') as f:
        for row_idx, line in enumerate(f, start=1):
            # 去除行尾换行符，并按空格分割（连续空格会被 split() 正确处理）
            parts = line.strip().split()
            col = 1
            for i in parts:
                ws.cell(row=row, column=col, value=i)
                col += 1
            row += 1
    wb.save(excel_path)
# 使用示例
if __name__ == "__main__":
    # 请根据实际情况修改文件路径
    input_txt = "/data/seery/src/log/reslut/memory/memory.txt"      # 输入的文本文件
    output_excel = "/data/seery/src/log/reslut/memory/memory.xlsx" # 输出的 Excel 文件

    txt_to_excel(input_txt, output_excel)