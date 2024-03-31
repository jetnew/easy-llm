from bs4 import BeautifulSoup

# Assuming `html_content` is the HTML content provided above
with open("data/cleaned.html", "r") as f:
    html_content = f.read()
# html_content = """<html> ... </html>"""  # The HTML content goes here

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Find the section containing experiences
experience_section = soup.find('section', string=lambda text: "Experience" in text if text else False)

# Assuming the latest experience is the first <li> element under the experience section
latest_experience = experience_section.find('li') if experience_section else None

# Extract the required information
if latest_experience:
    title = latest_experience.find_all('span')[0].get_text(strip=True)
    company = latest_experience.find_all('span')[1].get_text(strip=True)
    duration = latest_experience.find_all('span')[2].get_text(strip=True)
    
    # Prepare the result dictionary
    result = {
        "latest_experience_title": title,
        "latest_experience_company": company,
        "latest_experience_duration": duration,
    }
else:
    result = {
        "latest_experience_title": None,
        "latest_experience_company": None,
        "latest_experience_duration": None,
    }

# Print or use the result dictionary as needed
print(result)