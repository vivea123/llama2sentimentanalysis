
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.ticker as mticker
import tikzplotlib

def format_ticks(x, pos):
    return f'{x:,.0f}'

# Using SciencePlots style for the plots
# style.use('science')

# Define the directory containing the CSV files
directory = r"C:\Users\User\OneDrive - Lund University\Kandidatuppsats\Data\Headlines\Processed headlines"  # Replace with the actual path

# Initialize a dictionary to store headline counts per ticker
headline_counts_per_ticker = {}

# Initialize a dictionary to store headline counts per year
headline_counts_per_year = {}
total_headlines_count = 0

# Process each CSV file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        ticker = filename.split('.')[0]  # Extract ticker name from the file name
        file_path = os.path.join(directory, filename)

        # Read the CSV file and filter for dates from 2019 onwards
        df = pd.read_csv(file_path, usecols=['Date', 'Headline'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Date'].dt.year >= 2019]

        # Count headlines for this ticker and update the dictionary
        headline_counts_per_ticker[ticker] = len(df)

        # Count headlines per year
        total_headlines_count += len(df)

        # Count headlines for this ticker and update the dictionary
        headline_counts_per_ticker[ticker] = len(df)

        # Count headlines per year
        yearly_counts = df.groupby(df['Date'].dt.year).size()
        for year, count in yearly_counts.items():
            headline_counts_per_year[year] = headline_counts_per_year.get(year, 0) + count

print(total_headlines_count)

primary_color = "#22458a"
secondary_color = "#875e29"

# Plotting the total number of headlines per ticker
plt.figure(figsize=(10, 6))
plt.bar(headline_counts_per_ticker.keys(), headline_counts_per_ticker.values(), color=primary_color)
plt.xlabel('Ticker')
plt.ylabel('Number of Headlines')
plt.title('Total Number of Headlines per Ticker')
plt.xticks(rotation=45)
plt.tight_layout()
tikzplotlib.save(r"C:\Users\User\OneDrive - Lund University\Kandidatuppsats\Data\Headlines\Processed headlines\final_headlines_per_company.tex")

def format_func(value, tick_number):
    # Format the tick label as an integer without commas
    return f"{int(value)}"

def thousands_formatter(x, pos):
    # Format the y-axis as plain numbers (without k, M, etc.)
    return f'{int(x)}'

# Your data here
years = [2019, 2020, 2021, 2022, 2023]
headline_counts = [7536, 8516, 10433, 13998, 20388]

plt.figure(figsize=(10, 6))
plt.bar(years, headline_counts, color=primary_color)

# Formatting x-axis for years without commas#
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(format_func))

# Formatting y-axis to display full numbers
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(thousands_formatter))
plt.gca().tick_params(axis='y', which='major', pad=15)  # Increase pad value as needed

# Adjusting labels
plt.xlabel('Year', labelpad=15)  # Adjust labelpad as needed
plt.ylabel('Number of Headlines')

# Title and layout
plt.title('Total Number of Headlines per Year (All Companies)')
plt.tight_layout()

# Save using tikzplotlib
tikzplotlib.save(r"C:\Users\User\OneDrive - Lund University\Kandidatuppsats\Data\Headlines\Processed headlines\final_headlines_per_year_fixed.tex")