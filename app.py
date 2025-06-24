# app.py

import streamlit as st
import fitz  # PyMuPDF
import os
import pandas as pd
import re
import string
import time
from io import BytesIO
from tqdm import tqdm
from openai import AzureOpenAI

# === SETUP ===
st.set_page_config(page_title="Agri-Policy Classifier", layout="wide")

client = AzureOpenAI(
    api_key=st.secrets["AZURE_API_KEY"],
    api_version="2025-01-01-preview",
    azure_endpoint=st.secrets["AZURE_ENDPOINT"]
)
deployment = "gpt-4o"

theme_codebook = {
    "Farmer Welfare": ["Farmer Incomes", "Nutrition Security", "Health & Life Insurance", "Livelihood Security", "Costs of Cultivation"],
    "Production Support": ["Water Management", "Soil Health Management", "Fertilisers", "Pest Management", "Access to Credit"],
    "Price Support": ["Price Discovery", "Price Stability", "Fair and Assured Price", "Marketing", "Collectivisation"],
    "Risk Management": ["Crop Insurance", "Contract Farming"],
    "Environmental Sustainability": ["Climate Change", "Sustainable Agriculture", "Crop Diversification", "Central Government Schemes", "State Government Schemes"],
    "Financing": ["Union Budget Allocation", "State Budget Allocation", "Private Financing", "Climate Financing"]
}
main_themes = list(theme_codebook.keys())

# === UTILITIES ===
def clean_text(text):
    return ''.join(c for c in text if c in string.printable)

def is_probable_table(text):
    lines = text.split('\n')
    if "Table" in text or "Figure" in text:
        return True
    numeric_lines = sum(1 for line in lines if len(re.findall(r'\d+', line)) > 3)
    num_ratio = sum(c.isdigit() for c in text) / max(1, len(text))
    return numeric_lines >= 2 or num_ratio > 0.4

def extract_paragraphs_from_pdf(file):
    START_MARKERS = [r"^\s*(1|I|i)\.?\s*(Introduction|Executive Summary|Context|Overview|Background|Preamble)\b",
                     r"^\s*(Chapter|Section)\s+(1|I|i)\b", r"^\s*Main\s+Report\b", r"^\s*Agricultural\s+Subsidies\b"]
    END_MARKERS = [r"\breferences\b", r"\bbibliography\b", r"\bappendix\b"]

    def page_has_marker(text, patterns):
        return any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in patterns)

    def is_section_heading(text):
        return bool(re.match(r'^\s*\d+[\.\)]?\s+[A-Z].{3,}', text.strip()))

    doc = fitz.open(stream=file.read(), filetype="pdf")
    paragraphs, found_start = [], False

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not found_start:
            lines = text.splitlines()
            if page_has_marker(text, START_MARKERS) or any(is_section_heading(ln) for ln in lines):
                found_start = True
            else:
                continue
        if page_has_marker(text, END_MARKERS):
            break
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        chunks = re.split(r'\n{2,}', text)
        for chunk in chunks:
            para = clean_text(chunk.strip())
            if para and not is_probable_table(para):
                if re.match(r'^\d+[\.\)]?\s+[A-Z]', para) or re.match(r'^Figure\s+\d+', para) or re.match(r'^Table\s+\d+', para) or (40 <= len(para.split()) <= 1000):
                    paragraphs.append({"Document Name": file.name, "Page Number": page_num, "Paragraph": para})
    return paragraphs

def format_prompt(paragraph):
    return f"""You are an expert in agricultural sustainability and rural policy research.

Your task is to read the paragraph and do the following:

---

1️⃣ Select exactly **one main theme** from this list:
{', '.join(main_themes)}

2️⃣ Then, choose one or more **sub-themes strictly from the sub-theme list corresponding to the selected main theme**.

---

📚 Sub-theme list by main theme:

Farmer Welfare → Farmer Incomes, Nutrition Security, Health & Life Insurance, Livelihood Security, Costs of Cultivation  
Production Support → Water Management, Soil Health Management, Fertilisers, Pest Management, Access to Credit  
Price Support → Price Discovery, Price Stability, Fair and Assured Price, Marketing, Collectivisation  
Risk Management → Crop Insurance, Contract Farming  
Environmental Sustainability → Climate Change, Sustainable Agriculture, Crop Diversification, Central Government Schemes, State Government Schemes  
Financing → Union Budget Allocation, State Budget Allocation, Private Financing, Climate Financing

---

3️⃣ Write a brief but **insightful research summary**.
- Do NOT start with “This paragraph” or “It discusses”.
- Begin directly with the key insight.

---

📥 Return in this format:

Main Theme: <selected main theme>  
Sub-Themes: [comma-separated list]  
Summary: <research-usable summary>

Paragraph:
\"\"\"{paragraph}\"\"\"
"""

def safe_theme_name(theme):
    return re.sub(r"[^\w\s-]", "", str(theme)).strip().replace(" ", "_")

def classify_paragraphs(df):
    df["Main Theme"] = ""
    df["Sub-Theme(s)"] = ""
    df["Research Summary"] = ""
    for i in tqdm(df.index):
        prompt = format_prompt(df.at[i, "Paragraph"])
        try:
            res = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            ).choices[0].message.content
        except Exception as e:
            res = f"ERROR: {e}"
        try:
            df.at[i, "Main Theme"] = safe_theme_name(res.split("Main Theme:")[1].split("Sub-Themes:")[0].strip())
            subs = res.split("Sub-Themes:")[1].split("Summary:")[0].strip()
            df.at[i, "Sub-Theme(s)"] = ", ".join([s.strip() for s in subs.strip("[]").split(",")])
            df.at[i, "Research Summary"] = res.split("Summary:")[1].strip()
        except Exception as e:
            df.at[i, "Main Theme"] = "PARSE_ERROR"
            df.at[i, "Sub-Theme(s)"] = f"PARSE ERROR: {e}"
            df.at[i, "Research Summary"] = res
        time.sleep(1.5)
    return df

# === UI ===
st.title("📄 Agri-Policy Paragraph Classifier")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files and st.button("🚀 Run Classification"):
    os.makedirs("outputs/themes", exist_ok=True)
    all_results = []
    for file in uploaded_files:
        st.write(f"🔍 Extracting from: `{file.name}`")
        paragraphs = extract_paragraphs_from_pdf(file)
        df = pd.DataFrame(paragraphs)
        if not df.empty:
            df = classify_paragraphs(df)
            all_results.append(df)
        else:
            st.warning(f"⚠️ No paragraphs found in {file.name}")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        st.session_state.combined_df = combined_df  # persist
        combined_df.to_excel("outputs/Combined_Results.xlsx", index=False)
        st.success("✅ Results saved in `outputs/Combined_Results.xlsx`")

# === Show + Download ===
if "combined_df" in st.session_state:
    # st.subheader("📊 Classified Paragraphs")
    # st.dataframe(st.session_state.combined_df, use_container_width=True)

    # Combined download
    to_download = BytesIO()
    st.session_state.combined_df.to_excel(to_download, index=False)
    st.download_button("📥 Download Combined Excel", to_download.getvalue(), "Combined_Results.xlsx")

    # Theme-wise
    for theme in st.session_state.combined_df["Main Theme"].dropna().unique():
        theme_df = st.session_state.combined_df[st.session_state.combined_df["Main Theme"] == theme]
        if not theme_df.empty:
            safe_name = safe_theme_name(theme)
            path = f"outputs/themes/{safe_name}.xlsx"
            theme_df.to_excel(path, index=False)
            io_obj = BytesIO()
            theme_df.to_excel(io_obj, index=False)
            st.download_button(f"📂 Download: {safe_name}", io_obj.getvalue(), f"{safe_name}.xlsx")


# # app.py

# import streamlit as st
# import fitz  # PyMuPDF
# import os
# import pandas as pd
# import re
# import string
# import time
# from io import BytesIO
# from tqdm import tqdm
# from openai import AzureOpenAI

# # 🔐 Azure OpenAI setup
# client = AzureOpenAI(
#     api_key=st.secrets["AZURE_API_KEY"],
#     api_version="2025-01-01-preview",
#     azure_endpoint=st.secrets["AZURE_ENDPOINT"]
# )
# deployment = "gpt-4o"

# theme_codebook = {
#     "Farmer Welfare": ["Farmer Incomes", "Nutrition Security", "Health & Life Insurance", "Livelihood Security", "Costs of Cultivation"],
#     "Production Support": ["Water Management", "Soil Health Management", "Fertilisers", "Pest Management", "Access to Credit"],
#     "Price Support": ["Price Discovery", "Price Stability", "Fair and Assured Price", "Marketing", "Collectivisation"],
#     "Risk Management": ["Crop Insurance", "Contract Farming"],
#     "Environmental Sustainability": ["Climate Change", "Sustainable Agriculture", "Crop Diversification", "Central Government Schemes", "State Government Schemes"],
#     "Financing": ["Union Budget Allocation", "State Budget Allocation", "Private Financing", "Climate Financing"]
# }
# main_themes = list(theme_codebook.keys())

# START_MARKERS = [
#     r"^\s*(1|I|i)\.?\s*(Introduction|Executive Summary|Context|Overview|Background|Preamble)\b",
#     r"^\s*(Chapter|Section)\s+(1|I|i)\b",
#     r"^\s*Main\s+Report\b",
#     r"^\s*Agricultural\s+Subsidies\b"
# ]
# END_MARKERS = [r"\breferences\b", r"\bbibliography\b", r"\bappendix\b"]

# def clean_text(text):
#     return ''.join(char for char in text if char in string.printable)

# def is_probable_table(text):
#     lines = text.split('\n')
#     if "Table" in text or "Figure" in text:
#         return True
#     numeric_lines = sum(1 for line in lines if len(re.findall(r'\d+', line)) > 3)
#     num_ratio = sum(c.isdigit() for c in text) / max(1, len(text))
#     return numeric_lines >= 2 or num_ratio > 0.4

# def page_has_marker(text, patterns):
#     return any(re.search(p, text, re.IGNORECASE | re.MULTILINE) for p in patterns)

# def is_section_heading(text):
#     return bool(re.match(r'^\s*\d+[\.\)]?\s+[A-Z].{3,}', text.strip()))

# def extract_paragraphs_from_pdf(file):
#     doc = fitz.open(stream=file.read(), filetype="pdf")
#     paragraphs = []
#     found_start = False

#     for page_num, page in enumerate(doc, start=1):
#         text = page.get_text("text")

#         if not found_start:
#             lines = text.splitlines()
#             if page_has_marker(text, START_MARKERS) or any(is_section_heading(ln) for ln in lines):
#                 found_start = True
#             else:
#                 continue

#         if page_has_marker(text, END_MARKERS):
#             break

#         text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
#         chunks = re.split(r'\n{2,}', text)

#         for chunk in chunks:
#             para = clean_text(chunk.strip())
#             if not para:
#                 continue

#             important_line = (
#                 re.match(r'^\d+[\.\)]?\s+[A-Z]', para) or
#                 re.match(r'^Figure\s+\d+', para) or
#                 re.match(r'^Table\s+\d+', para) or
#                 40 <= len(para.split()) <= 1000
#             )

#             if important_line and not is_probable_table(para):
#                 paragraphs.append({
#                     "Document Name": file.name,
#                     "Page Number": page_num,
#                     "Paragraph": para
#                 })
#     return paragraphs

# def format_prompt(paragraph):
#     return f"""You are an expert in agricultural sustainability and rural policy research.

# Your task is to read the paragraph and do the following:

# ---

# 1️⃣ Select exactly **one main theme** from this list:
# {', '.join(main_themes)}

# 2️⃣ Then, choose one or more **sub-themes strictly from the sub-theme list corresponding to the selected main theme**.

# ---

# 📚 Sub-theme list by main theme:

# Farmer Welfare → Farmer Incomes, Nutrition Security, Health & Life Insurance, Livelihood Security, Costs of Cultivation  
# Production Support → Water Management, Soil Health Management, Fertilisers, Pest Management, Access to Credit  
# Price Support → Price Discovery, Price Stability, Fair and Assured Price, Marketing, Collectivisation  
# Risk Management → Crop Insurance, Contract Farming  
# Environmental Sustainability → Climate Change, Sustainable Agriculture, Crop Diversification, Central Government Schemes, State Government Schemes  
# Financing → Union Budget Allocation, State Budget Allocation, Private Financing, Climate Financing

# ---

# 3️⃣ Write a brief but **insightful research summary**.
# - Do NOT start with “This paragraph” or “It discusses”.
# - Begin directly with the key insight.

# ---

# 📥 Return in this format:

# Main Theme: <selected main theme>  
# Sub-Themes: [comma-separated list]  
# Summary: <research-usable summary>

# Paragraph:
# \"\"\"{paragraph}\"\"\"
# """

# def safe_theme_name(theme):
#     return re.sub(r"[^\w\s-]", "", str(theme)).strip().replace(" ", "_")

# def classify_paragraphs(df):
#     df["Main Theme"] = ""
#     df["Sub-Theme(s)"] = ""
#     df["Research Summary"] = ""

#     for i in tqdm(df.index, desc="Classifying"):
#         paragraph = df.at[i, "Paragraph"]
#         prompt = format_prompt(paragraph)
#         try:
#             response = client.chat.completions.create(
#                 model=deployment,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0
#             )
#             result = response.choices[0].message.content
#         except Exception as e:
#             result = f"ERROR: {e}"

#         try:
#             main_theme = result.split("Main Theme:")[1].split("Sub-Themes:")[0].strip()
#             subthemes_raw = result.split("Sub-Themes:")[1].split("Summary:")[0].strip()
#             summary = result.split("Summary:")[1].strip()
#             subtheme_list = [s.strip() for s in subthemes_raw.strip("[]").split(",") if s.strip()]
#             df.at[i, "Main Theme"] = safe_theme_name(main_theme)
#             df.at[i, "Sub-Theme(s)"] = ", ".join(subtheme_list)
#             df.at[i, "Research Summary"] = summary
#         except Exception as e:
#             df.at[i, "Research Summary"] = result
#             df.at[i, "Main Theme"] = "PARSE_ERROR"
#             df.at[i, "Sub-Theme(s)"] = f"PARSE ERROR: {e}"

#         time.sleep(1.5)
#     return df

# # === Streamlit UI ===
# st.title("📄 Agri-Policy Paragraph Classifier")
# st.markdown("Upload one or more policy PDFs to extract and classify meaningful paragraphs using Azure GPT.")

# uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# if uploaded_files and st.button("🚀 Run Classification"):
#     all_results = []

#     for file in uploaded_files:
#         st.write(f"📂 Processing file: `{file.name}`")
#         extracted = extract_paragraphs_from_pdf(file)
#         df = pd.DataFrame(extracted)

#         if not df.empty:
#             df = classify_paragraphs(df)
#             all_results.append(df)
#         else:
#             st.warning(f"No valid paragraphs found in: {file.name}")

#     if all_results:
#         combined_df = pd.concat(all_results, ignore_index=True)
#         combined_df.to_excel("Combined_Results.xlsx", index=False)
#         st.success("✅ Combined Excel generated!")

#         with open("Combined_Results.xlsx", "rb") as f:
#             st.download_button("📥 Download Combined Excel", f, file_name="Combined_Results.xlsx")

#         # Theme-wise Excel outputs
#         os.makedirs("themes_outputs", exist_ok=True)
#         for theme in combined_df["Main Theme"].dropna().unique():
#             theme_df = combined_df[combined_df["Main Theme"] == theme]
#             if not theme_df.empty:
#                 file_path = os.path.join("themes_outputs", f"{safe_theme_name(theme)}.xlsx")
#                 theme_df.to_excel(file_path, index=False)
#                 with open(file_path, "rb") as f:
#                     st.download_button(f"📂 Download: {safe_theme_name(theme)}", f, file_name=os.path.basename(file_path))
#     else:
#         st.error("⚠️ No classified data to export.")

