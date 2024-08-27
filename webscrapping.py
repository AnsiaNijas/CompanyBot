import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from fpdf import FPDF
import time

# Function to fetch and parse website content using Selenium
def fetch_website_content_with_selenium(url):
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(5)
    content = driver.find_element(By.TAG_NAME, "body").text
    driver.quit()
    return content

def save_as_pdf(content, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    # Encode Unicode content properly for PDF
    for line in content.split('\n'):
        # Handling basic Unicode by encoding/decoding
        line = line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)

if __name__ == "__main__":
    url = os.getenv("url")
    content = fetch_website_content_with_selenium(url)
    pdf_filename = os.getenv("pdf_filename")
    save_as_pdf(content, pdf_filename)
    print(f"Website content saved as {pdf_filename}")
