"""Author: Rashid Jama Abdi
Section Leader: Sara
Date: 11/18/24
ISTA 350 final Project
The projects is about webscraping three nba links and taking the Advance 
table in the website, the table includes stats about the players, and 
in the code below I will be creating three visulizations, 
Radar chart, bar Chart and scatter plot with linear regression lines:"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to extract table data
def extract_table_data(soup, table_id):
    table = soup.find('table', {'id': table_id})
    if table:
        headers = [th.getText() for th in table.find('thead').find_all('th')]
        rows = table.find('tbody').find_all('tr')
        data = []
        for row in rows:
            cells = row.find_all('td')
            if cells:
                data.append([cell.getText() for cell in cells])
        return headers, data
    else:
        return None, None

# Function to create a DataFrame from a URL
def create_dataframe_from_url(url, table_id):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headers, data = extract_table_data(soup, table_id)
    if headers and data:
        return pd.DataFrame(data, columns=headers[1:])
    else:
        print(f"Table not found or data is empty for URL: {url}")
        return pd.DataFrame()
# Funtion to create player radar charts
def radar_chart(df, selected_players):
    metrics = ['PER', 'TS%', 'AST%', 'TRB%', 'STL%']
    categories = metrics
    N = len(categories)
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
    for player in selected_players:
        player_data = df[df['Player'] == player][metrics].copy()
        print(f"\nData for {player}:")
        print(player_data)
        if player_data.isnull().values.any():
            print(f"Missing data for {player}. Skipping chart.")
            continue
        global_max = df[metrics].max()
        global_min = df[metrics].min()
        normalized_data = (player_data - global_min) / (global_max - global_min)
        player_stats = normalized_data.values.flatten().tolist()
        player_stats += player_stats[:1]
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6, 6))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color='grey', size=8)
        ax.plot(angles, player_stats, linewidth=1, linestyle='solid', label=player)
        ax.fill(angles, player_stats, alpha=0.1)
        ax.set_title(player, size=11, color='blue', y=1.1)
        # Show plot
        plt.tight_layout()
        plt.show()

# Function to create grouped bar chart for team comparison
def team_comparison_bar_chart(df):
    metrics = ['PER', 'BPM', 'WS/48']
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
    team_averages = df.groupby('Team')[metrics].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.25
    r1 = range(len(team_averages))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    ax.bar(r1, team_averages['PER'], color='b', width=bar_width, edgecolor='grey', label='PER')
    ax.bar(r2, team_averages['BPM'], color='g', width=bar_width, edgecolor='grey', label='BPM')
    ax.bar(r3, team_averages['WS/48'], color='r', width=bar_width, edgecolor='grey', label='WS/48')
    ax.set_xlabel('Team', fontweight='bold')
    ax.set_ylabel('Average Values', fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(team_averages))])
    ax.set_xticklabels(team_averages['Team'])
    plt.legend()
    plt.title('Team Performance Comparison')
    plt.tight_layout()
    plt.show()

# Function to calculate Pearson correlation coefficient
def calculate_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return numerator / denominator

# Function to create separate scatter plots with linear regression for each team
def usage_vs_efficiency_scatter_plot(df):
    df['USG%'] = pd.to_numeric(df['USG%'], errors='coerce')
    df['TS%'] = pd.to_numeric(df['TS%'], errors='coerce')
    df = df.dropna(subset=['USG%', 'TS%'])

    teams = df['Team'].unique()
    colors = {'GSW': 'blue', 'PHO': 'orange', 'LAL': 'purple'}

    for team in teams:
        team_data = df[df['Team'] == team]
        plt.figure(figsize=(8, 6))
        plt.scatter(team_data['USG%'], team_data['TS%'], c=colors[team], label=team, alpha=0.6, edgecolors='w', s=100)
        X = team_data[['USG%']].values
        y = team_data['TS%'].values
        model = LinearRegression().fit(X, y)
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = model.predict(X_range)
        plt.plot(X_range, y_pred, color='black', linewidth=2)
        # Calculate and display correlation
        correlation = calculate_correlation(X.flatten(), y)
        plt.title(f'Usage vs. Efficiency for {team}\nCorrelation: r={correlation:.2f}')
        plt.xlabel('Usage Rate (USG%)')
        plt.ylabel('True Shooting Percentage (TS%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



def main():
    urls = [
        'https://www.basketball-reference.com/teams/GSW/2025.html',
        'https://www.basketball-reference.com/teams/PHO/2025.html',
        'https://www.basketball-reference.com/teams/LAL/2025.html'
    ]

    # Table ID for the "Advanced Stats" table
    table_id = 'advanced'
    all_dataframes = []
    for url in urls:
        team_name = url.split('/')[-2]
        df = create_dataframe_from_url(url, table_id)
        if not df.empty:
            df['Team'] = team_name
            all_dataframes.append(df)
    # Combine all DataFrames into one
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(combined_df)

    # Define selected players for radar chart
    selected_players = ['Stephen Curry', 'Kevin Durant', 'LeBron James']

    # Generate radar chart
    radar_chart(combined_df, selected_players)

    # Generate team comparison bar chart
    team_comparison_bar_chart(combined_df)

    # Generate scatter plot
    usage_vs_efficiency_scatter_plot(combined_df)

if __name__ == "__main__":
    main()