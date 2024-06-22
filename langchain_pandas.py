from langchain.agents.agent_types import AgentType

from langchain_community.chat_models import ChatOllama

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import pandas as pd

url_df = 'https://fbref.com/en/comps/676/stats/European-Championship-Stats'
df = pd.read_html(url_df)[0]

df.columns = [' '.join(col).strip() for col in df.columns]
df = df.reset_index(drop=True)
new_columns = []
for col in df.columns:
    if "level_0" in col:
        new_col = col.split()[-1]  # takes the last name
    else:
        new_col = col
    new_columns.append(new_col)
df.columns = new_columns
df = df.fillna(0)

df[['Squad Abbreviation', 'Team']] = df['Squad'].str.split(' ', expand=True)
df.drop(['Squad', 'Squad Abbreviation'], axis=1, inplace=True)
df.rename(columns={'Team': 'Squad',
                   'Pl': 'Number of players used in games',
                   'Age': 'Average age is weighted by minutes played',
                   'Poss': 'Possession calculated as the percentage of passes attempted',
                   'Playing Time MP': 'Matches Played by the player or squad',
                   'Playing Time Starts': 'Game or games started by player',
                   'Playing Time Min': 'Minutes',
                   'Playing Time 90s': 'Minutes played divided by 90',
                   'Performance Gls': 'Goals scored or allowed',
                   'Performance Ast': 'Assists',
                   'Performance G+A': 'Goals and assists',
                   'Performance G-PK': 'Non-Penalty goals',
                   'Performance PK': 'Penalty kicks made',
                   'Performance PKatt': 'Penalty kicks attempted',
                   'Performance CrdY': 'Yellow cards',
                   'Performance CrdR': 'Red cards',
                   'Expected xG': 'Expected goals totals include penalty kicks, but do not include penalty shootouts (unless otherwise noted).',
                   'Expected npxG': 'Non-Penalty Expected Goals',
                   'Expected xAG': 'Expected assisted goals which follows a pass that assists a shot',
                   'Expected npxG+xAG': 'Non-Penalty expected goals plus assisted goals',
                   'Progression PrgC': 'Progressive Carries that move the ball towards the opponent\'s goal line at least 10 yards from its furthest point in the last six passes, or any carry into the penalty area. Excludes carries which end in the defending 50% of the pitch',
                   'Progression PrgP': 'Progressive passes. Completed passes that move the ball towards the opponent\'s goal line at least 10 yards from its furthest point in the last six passes, or any completed pass into the penalty area. Excludes passes from the defending 40% of the pitch per 90 Minutes',
                   'Per 90 Minutes Gls': 'Goals Scored per 90 minutes',
                   'Per 90 Minutes Ast': 'Assists per 90 minutes',
                   'Per 90 Minutes G+A': 'Goals and Assists per 90 minutes',
                   'Per 90 Minutes G-PK': 'Goals minus Penalty Kicks made per 90 minutes',
                   'Per 90 Minutes G+A-PK': 'Goals plus Assists minus Penalty Kicks made per 90 minutes',
                   'Per 90 Minutes xG': 'Expected Goals per 90 minutes. xG totals include penalty kicks, but do not include penalty shootouts (unless otherwise noted).',
                   'Per 90 Minutes xAG': 'Expected Assisted Goals per 90 minutes',
                   'Per 90 Minutes xG+xAG': 'Expected Goals plus Assisted Goals per 90 minutes',
                   'Per 90 Minutes npxG': 'Non-Penalty Expected Goals per 90 minutes',
                   'Per 90 Minutes npxG+xAG': 'Non-Penalty Expected Goals plus Assisted Goals per 90 minutes'
                   }, inplace=True)

print(df.head())

agent = create_pandas_dataframe_agent(
    ChatOllama(temperature=0, model="llama3"),
    df,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    allow_dangerous_code=True,
)

# agent.invoke({"input": "How many rows are there?"})
# agent.invoke({"input": "What Squad has the most expected goals?"})
# agent.invoke({"input": "Is Türkiye or Portugal the better team?"})
agent.invoke({"input": "How will be exact result of a soccer game between Belgium or Romania?"})
