# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:33:02 2022

@author: Usuario

"""


# In[2]:

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori
#import re
#from stockfish import Stockfish
#import chess
#import chess.engine
#import math


#Stockfish 15
#engine = chess.engine.SimpleEngine.popen_uci(r"C:\Users\Usuario\Downloads\stockfish_15_win_x64_avx2\stockfish_15_win_x64_avx2")
#stockfish = Stockfish(path = r'C:\Users\Usuario\Downloads\stockfish_15_win_x64_avx2\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe')
#stockfish.set_elo_rating(2600)
df = pd.read_csv('chess.csv')


#print(df.dtypes)
#df.head(1)

#Partides amb pocs moviments
df = df[df.turns >= 6]


#Making a new Series in the dataframe to contain a list of all the moves, since the move list as a string is not all that useful to us
df['moves_list'] = df.moves.apply(lambda x: x.split())

#print(df['moves_list'])


#All white moves and all black moves



#Add columns called opening_move and response, which are the first moves by white and black, respectively

df['Apertura'] = df.moves_list.apply(lambda x: x[0])
df['response'] = df.moves_list.apply(lambda x: x[1])
df['Apertura1'] = df.moves_list.apply(lambda x: x[2])
df['response1'] = df.moves_list.apply(lambda x: x[3])
df['Apertura2'] = df.moves_list.apply(lambda x: x[4])
df['response2'] = df.moves_list.apply(lambda x: x[5])


df['3W'] = df['Apertura'] + ' ' + df['Apertura1'] + ' ' + df['Apertura2']
df['3B'] = df['response'] + ' ' + df['response1'] + ' ' + df['response']

df['3First'] = df['Apertura'] + ' ' + df['response'] + ' ' + df['Apertura1'] + ' ' + df['response1']+ ' ' + df['Apertura2']+ ' ' + df['Apertura2']

print(df['3W'])

def most_common(df):
    return max(set(df), key=df.count)

  # In[2]:  
    

#Adding a column for opening/response name
df['opening_name'] = df.moves_list.apply(lambda x: 'Peó de Rei (E4)' if x[0] == 'e4' else ('Peó de Reina(D4)' if x[0] == 'd4' else ('Anglesa' if x[0] == 'c4' else ('Reti' if x[0] == 'Nf3' else ('Fianchetto' if x[0] == 'g6' else 'Altres')))))
df['response_name'] = df.moves_list.apply(lambda x: 'Peó de Rei (E5)' if x[1] == 'e5' else ('Peó de Reina(D5)' if x[1] == 'd5' else ('Siciliana' if x[1] == 'c5' else ('Defensa Francesa' if x[1] == 'e6' else 'Altres'))))

#Rating Gap
df['rating_gap'] = abs(df['white_rating'] - df['black_rating'])

#Did the higher-rated player win?
df['higher_rated_victory'] = np.where((df['winner'] == 'White') & (df['white_rating'] > df['black_rating']) | (df['winner'] == 'Black') & (df['black_rating'] > df['white_rating']), 1, 0)

#Classify the rating gap. 
df['rating_gap_class'] = df.rating_gap.apply(lambda x: '0-50' if (x <= 50) else ('51-100' if (x > 50 and x <= 100) else ('101-150' if (x > 100 and x <= 150) else ('151-200' if (x > 150 and x <= 200) else ('201-250' if (x > 200 and x <= 250) else '>250')))))

#Did white win?
df['white_victory'] = np.where(df['winner'] == 'White', 1, 0)




#5 openings for white
english = df[df.Apertura == 'c4']
queens_pawn = df[df.Apertura == 'd4']
kings_pawn = df[df.Apertura == 'e4']
reti = df[df.Apertura == 'Nf3']
fianchetto = df[df.Apertura == 'g3']

#5 opening by black
kings_pawn1=df[df.response == 'e5']
queens_pawn1= df[df.response == 'd5']
french= df[df.response == 'e6']
siciliana= df[df.response == 'c5']


#Visualize which openings are used the most
opening_data = df.groupby('opening_name')['game_id'].count()
plt.pie(x=opening_data, autopct='%.1f%%', labels=opening_data.keys(), pctdistance=0.5)
plt.title('Percentatge apertures', fontsize=14)
plt.axis('equal') #ensures pie chart is a circle
plt.show()

#Visualize which responses are used the most
response_data = df.groupby('response_name')['game_id'].count()
plt.pie(x=response_data, autopct='%.1f%%', labels=response_data.keys(), pctdistance=0.5)
plt.title('Percentatge respostes', fontsize=14)
plt.axis('equal') #ensures pie chart is a circle
plt.show()


#Visualize who was able to win after each opening using pie chart

#For each DataFrame, create additional DataFrame counting the number of times white or black won (or drew)
english_winner_data = english.groupby('winner')['game_id'].count()
queen_winner_data = queens_pawn.groupby('winner')['game_id'].count()
king_winner_data = kings_pawn.groupby('winner')['game_id'].count()
reti_winner_data = reti.groupby('winner')['game_id'].count()
fianchetto_winner_data = fianchetto.groupby('winner')['game_id'].count()


#For each DataFrame, create additional DataFrame counting the number of times white or black won (or drew)
french_winner_data = french.groupby('winner')['game_id'].count()
queen1_winner_data = queens_pawn1.groupby('winner')['game_id'].count()
king1_winner_data = kings_pawn1.groupby('winner')['game_id'].count()
siciliana_winner_data = siciliana.groupby('winner')['game_id'].count()

pie, axs = plt.subplots(2,2, figsize=[10,10])

plt.subplot(2,2,1)
plt.pie(x=english_winner_data, autopct="%.1f%%", labels=english_winner_data.keys(), pctdistance=0.5)
plt.title("Guanyar obrint amb C4", fontsize=8)
plt.axis('equal')


plt.subplot(2,2,2)
plt.pie(x=queen_winner_data, autopct="%.1f%%", labels=queen_winner_data.keys(), pctdistance=0.5)
plt.title("Guanyar obrint amb D4", fontsize=8)
plt.axis('equal')

plt.subplot(2,2,3)
plt.pie(x=king_winner_data, autopct="%.1f%%", labels=king_winner_data.keys(), pctdistance=0.5)
plt.title("Guanyar obrint amb E4", fontsize=8)
plt.axis('equal')

plt.subplot(2,2,4)
plt.pie(x=reti_winner_data, autopct="%.1f%%", labels=reti_winner_data.keys(), pctdistance=0.5)
plt.title("Guanyar obrint amb NF3", fontsize=8)
plt.axis('equal')

plt.show()

plt.subplot(3,3,5)
plt.pie(x=fianchetto_winner_data, autopct="%.1f%%", labels=fianchetto_winner_data.keys(), pctdistance=0.5)
plt.title("Guanyadors després d'obrir amb G6", fontsize=7)
plt.axis('equal')
plt.show()



pie, axs = plt.subplots(2,2, figsize=[10,10])

plt.subplot(2,2,1)
plt.pie(x=french_winner_data, autopct="%.1f%%", labels=french_winner_data.keys(), pctdistance=0.5)
plt.title("Guanyar defensant amb E6", fontsize=8)
plt.axis('equal')


plt.subplot(2,2,2)
plt.pie(x=queen1_winner_data, autopct="%.1f%%", labels=queen1_winner_data.keys(), pctdistance=0.5)
plt.title("Guanyar defensant amb D5", fontsize=8)
plt.axis('equal')

plt.subplot(2,2,3)
plt.pie(x=king1_winner_data, autopct="%.1f%%", labels=king1_winner_data.keys(), pctdistance=0.5)
plt.title("Guanyar defensant amb E5", fontsize=8)
plt.axis('equal')

plt.subplot(2,2,4)
plt.pie(x=siciliana_winner_data, autopct="%.1f%%", labels=siciliana_winner_data.keys(), pctdistance=0.5)
plt.title("Guanyar defensant amb C5", fontsize=8)
plt.axis('equal')

plt.show()

# In[3]:

#

#Counts the total amount of white piece victories, grouped by opening name
df_grouped_ratings = df.groupby('opening_name')['higher_rated_victory'].sum()
df_grouped_ratings = df_grouped_ratings.to_frame() #Converting from series to dataFrame

#Counts the total amount of black piece victories, grouped by response name
df_grouped_ratings1 = df.groupby('response_name')['higher_rated_victory'].sum()
df_grouped_ratings1 = df_grouped_ratings1.to_frame() #Converting from series to dataFrame


#Adding a column for the total number of games
df_grouped_ratings['totals'] = df.groupby('opening_name')['higher_rated_victory'].count()
#Adding a column for draws/losses
df_grouped_ratings['losses_or_draws'] = df_grouped_ratings['totals'] - df_grouped_ratings['higher_rated_victory']
print(df_grouped_ratings.head(10))


#Adding a column for the total number of games
df_grouped_ratings1['totals'] = df.groupby('response_name')['higher_rated_victory'].count()

#Adding a column for draws/losses
df_grouped_ratings1['losses_or_draws'] = df_grouped_ratings1['totals'] - df_grouped_ratings1['higher_rated_victory']
print(df_grouped_ratings1.head(10))


#Visualize which ratings dominate which opening

df_grouped_ratings = df_grouped_ratings.sort_values('totals', ascending=False)
fig, ax = plt.subplots(1, figsize=(12,10))
ax.bar([0,1,2,3,4], df_grouped_ratings['higher_rated_victory'], label='Gunayar a més ELO', color='#ae24d1', tick_label=df_grouped_ratings.index)
ax.bar([0,1,2,3,4], df_grouped_ratings['losses_or_draws'], label='Perdre o Empatar vs menys ELO', bottom=df_grouped_ratings['higher_rated_victory'], color='#24b1d1')
ax.set_ylabel('Victòries', fontsize=14)
ax.set_xlabel('Apertura', fontsize=14)
ax.set_title('Victòries per Apertura', fontsize=18)
ax.legend()
plt.show()

#Visualize which ratings dominate which response

df_grouped_ratings1 = df_grouped_ratings1.sort_values('totals', ascending=False)
fig, ax = plt.subplots(1, figsize=(12,10))
ax.bar([0,1,2,3,4], df_grouped_ratings1['higher_rated_victory'], label='Gunayar a més ELO', color='#ae24d1', tick_label=df_grouped_ratings1.index)
ax.bar([0,1,2,3,4], df_grouped_ratings1['losses_or_draws'], label='Perdre o Empatar vs menys ELO', bottom=df_grouped_ratings1['higher_rated_victory'], color='#24b1d2')
ax.set_ylabel('Victòries', fontsize=14)
ax.set_xlabel('Resposta', fontsize=14)
ax.set_title('Victòries per resposta', fontsize=18)
ax.legend()
plt.show()


pie, axs = plt.subplots(2,2, figsize=[16,10])

plt.subplot(2,2,1)
sns.barplot(
    data=kings_pawn,
    x='response',
    y='white_victory',
    
)
plt.title('Resposta a Peó de Rei')
plt.ylabel('Winrate (Blanc)')
plt.xlabel('Resposta de Negres')
plt.axhline(0.5)

plt.subplot(2,2,2)
sns.barplot(
    data=queens_pawn,
    x='response',
    y='white_victory',
    palette='Spectral'
)
plt.title('Resposta a Peó de Reina')
plt.ylabel('Winrate (Blanc)')
plt.xlabel('Resposta de Negres')
plt.axhline(0.5)

plt.subplot(2,2,3)
sns.barplot(
    data=english,
    x='response',
    y='white_victory',
    palette='brg_r'
)
plt.title('Resposta a Anglesa')
plt.ylabel('Winrate (Blanc)')
plt.xlabel('Resposta de Negres')
plt.axhline(0.5)

plt.subplot(2,2,4)
sns.barplot(
    data=reti,
    x='response',
    y='white_victory',
    palette='Wistia'
)
plt.title('Resposta a Reti')
plt.ylabel('Winrate (Blanc)')
plt.xlabel('Resposta de Negres')
plt.axhline(0.5)

plt.show()


# In[4]:

#Create desired dataframe where every player is rated under a certain ELO


num = 1200
num1=1600
num2=2000
num3=2400
num4=2800


df_under_num = df[(df.white_rating < num) & (df.black_rating < num)]

df_under_num_winners = df_under_num.groupby('winner')['game_id'].count()
df_under_num_winners


df_under_num1 = df[(df.white_rating < num1) & (df.black_rating < num1)]

df_under_num1_winners = df_under_num1.groupby('winner')['game_id'].count()
df_under_num1_winners

df_under_num2 = df[(df.white_rating < num2) & (df.black_rating < num2)]

df_under_num2_winners = df_under_num2.groupby('winner')['game_id'].count()
df_under_num2_winners

df_under_num3 = df[(df.white_rating < num3) & (df.black_rating < num3)]

df_under_num3_winners = df_under_num3.groupby('winner')['game_id'].count()
df_under_num3_winners

df_under_num4 = df[(df.white_rating < num4) & (df.black_rating < num4)]

df_under_num4_winners = df_under_num4.groupby('winner')['game_id'].count()
df_under_num4_winners



pie, axs = plt.subplots(5,1, figsize=[18,28])

plt.subplot(5,1,1)
plt.pie(x=df_under_num_winners, autopct="%.1f%%", labels=df_under_num_winners.keys(), pctdistance=0.5)
plt.title("Winrate per sota {} ELO".format(num), fontsize=10)
plt.axis('equal')


plt.subplot(5,1,2)
plt.pie(x=df_under_num1_winners, autopct="%.1f%%", labels=df_under_num1_winners.keys(), pctdistance=0.5)
plt.title("Winrate per sota {} ELO".format(num1), fontsize=10)
plt.axis('equal')


plt.subplot(5,1,3)
plt.pie(x=df_under_num2_winners, autopct="%.1f%%", labels=df_under_num2_winners.keys(), pctdistance=0.5)
plt.title("Winrate per sota {} ELO".format(num2), fontsize=10)
plt.axis('equal')

plt.subplot(5,1,4)
plt.pie(x=df_under_num3_winners, autopct="%.1f%%", labels=df_under_num3_winners.keys(), pctdistance=0.5)
plt.title("Winrate per sota {} ELO".format(num3), fontsize=10)
plt.axis('equal')

plt.subplot(5,1,5)
plt.pie(x=df_under_num4_winners, autopct="%.1f%%", labels=df_under_num4_winners.keys(), pctdistance=0.5)
plt.title("Winrate per sota {} ELO".format(num4), fontsize=10)
plt.axis('equal')

plt.show()


sns.barplot(
    data=df,
    x='rating_gap_class',
    y='higher_rated_victory',
    order=['0-50', '51-100', '101-150', '151-200', '201-250', '>250'],
    
)
plt.xlabel('Diferència ELO', fontsize=16)
plt.ylabel('Win Rate per més ELO', fontsize=16)
plt.title('Win Rate per diferència ELO', fontsize=16)
plt.show()



#users who played white pieces and how many wins they accrued (DATAFRAME)
df_users_white = df.groupby('white_id',as_index=False).agg({'white_victory':'sum', 'game_id':'count'})
df_users_white.rename(columns={'white_victory':'white_victories', 'game_id':'white_games'}, inplace=True)
df_users_white = df_users_white.set_index('white_id')

#users who had black pieces and how many victories their opponents accrued (DATAFRAME)
df_users_black = df.groupby('black_id',as_index=False).agg({'white_victory':'sum', 'game_id':'count'})
df_users_black.rename(columns={'white_victory':'white_victories', 'game_id':'black_games'}, inplace=True)
df_users_black = df_users_black.set_index('black_id')

#Since we only know white piece victories, we need black piece victories to give us the number of games won by the key
df_users_black['black_victories'] = df_users_black.black_games - df_users_black.white_victories
df_users_black.drop('white_victories', axis=1, inplace=True)

#Join the two dataframes together on the index (username in this case). Some will have NaN values because of the nature of the join
#But we can just replace those with 0
df_users = df_users_white.join(df_users_black)
df_users = df_users.fillna(0) #Replace all NaN with 0 


#add columns for total victories and total games played
df_users['victories'] = df_users.white_victories + df_users.black_victories
df_users['games_played'] = df_users.white_games + df_users.black_games

#Win percentatge 
df_users['win_pct'] = df_users.victories / df_users.games_played



#Let's sort the df by people who have the highest win percentage
df_users_sorted = df_users.sort_values(by=['win_pct'], ascending=False)
df_users_sorted.head(5)


game_threshold = 24
df_users_many_games = df_users[(df_users.games_played >= game_threshold)]
df_users_many_games_sorted = df_users_many_games.sort_values(by=['win_pct'], ascending=False)
df_users_many_games_sorted.head(10)

print(df_users_many_games_sorted)






    
association_rules = apriori(df['3First'], min_support=0.10, min_confidence=0.2, min_lift=3, min_length=6)
#association_rules_black = apriori(df['3B'], min_support=0.063, min_confidence=0.2, min_lift=3, min_length=6)


# In[3]:
association_rules = apriori(df['3W'], min_support=0.10, min_confidence=0.2, min_lift=3, min_length=6)

for item in association_rules:
   pair = df['3W']
   items = [x for x in pair]
   print("Rule White: " + items[0] + items[1] + "  -> "  + (items[3]+ items[4])) #Item [2] és l'espai en blanc
   print("Support White: " + str(item[1]))
   print("Confidence White: " + str(item[2][0][2]))  #REVISAR FÒRMULES
   print("Lift White: " + str(item[2][0][3]))
   print("=====================================")
    
#for item in association_rules_black:
    #pair = df['3First'][0] 
    #items = [x for x in pair]
    #print("Rule Black " + items[0]+items[1] + "  -> "  + (items[3]+ items[4])) #Item [2] és l'espai en blanc
    #print("Support Black: " + str(item[1]))
    #print("Confidence Black: " + str(item[2][0][2])) #REVISAR FÒRMULES
    #print("Lift Black: " + str(item[2][0][3]))
    #print("=====================================")

# In[9]:

#Creem les regles d'associació i apliquem apriori per poder veure les combinacions més comunes

association_rules = apriori(df['3W'], min_support=0.1, min_confidence=0.2, min_lift=3.39, min_length=10)

for item in association_rules:
        pair = df['3W'][0]
        items = [x for x in pair]
        print("Rule White: " + items[0] + items[1] + " -> "  + (items[3]+ items[4]) + " -> " + (items[5]+ items[6] +items[7] + items[8] + items[9] )) #Item [2] és l'espai en blanc
        print("Support White: " + str(item[1]))
        print("Confidence White: " + str(item[2][0][2]))  
        print("Lift White: " + str(item[2][0][3]))
        print("=====================================")
        
        
# In[10]:

association_rules_black = apriori(df['3B'], min_support=0.1, min_confidence=0.2, min_lift=4.4, min_length=10)



for item in association_rules_black:
        pair = df['3B']
        items = [x for x in pair]
        print("Rule Black " + (items[0])) # items[1] + items[2])) #+ #items[3]) + (items[4]+ items[5])+ (items[6]+ items[7])) #Item [2] és l'espai en blanc
        print("Support Black: " + str(item[1]))
        print("Confidence Black: " + str(item[2][0][2])) 
        print("Lift Black: " + str(item[2][0][3]))
        print("=====================================")

# In[11]:
    
association_rules_first = apriori(df['3First'], min_support=0.075, min_confidence=0.2, min_lift=3, min_length=6)

for item in association_rules_first:
    pair = df['3First']
    items = [x for x in pair]
    print("Rule First " + items[0] + items[1] + items[2] + items[3] + items[4] + items[5] + items[6]) # + items[7] + items[8] + items[9] + items[10] + items[11] + items[12] + items[13] + items[14] + items[15] + items[16] + items[17] + items[18] ) #Item [2] és l'espai en blanc
    print("Support First: " + str(item[1]))
    print("Confidence First: " + str(item[2][0][2])) 
    print("Lift First: " + str(item[2][0][3]))
    print("=====================================")

# In[9]:

#Creem les regles d'associació i apliquem apriori per poder veure les combinacions més comunes

association_rules = apriori(df['3W'], min_support=0.1, min_confidence=0.2, min_lift=3.39, min_length=10)

for item in association_rules:
        pair = df['3W'][0]
        items = [x for x in pair]
        print("Rule White: " + items[0] + items[1] + " -> "  + (items[3]+ items[4]) + " -> " + (items[5]+ items[6] +items[7] + items[8] + items[9] )) #Item [2] és l'espai en blanc
        print("Support White: " + str(item[1]))
        print("Confidence White: " + str(item[2][0][2]))  
        print("Lift White: " + str(item[2][0][3]))
        print("=====================================")
        
        
# In[10]:

association_rules_black = apriori(df['3B'], min_support=0.1, min_confidence=0.2, min_lift=4.4, min_length=10)



for item in association_rules_black:
        pair = df['3B']
        items = [x for x in pair]
        print("Rule Black " + (items[0])) # items[1] + items[2])) #+ #items[3]) + (items[4]+ items[5])+ (items[6]+ items[7])) #Item [2] és l'espai en blanc
        print("Support Black: " + str(item[1]))
        print("Confidence Black: " + str(item[2][0][2])) 
        print("Lift Black: " + str(item[2][0][3]))
        print("=====================================")

# In[11]:
    
association_rules_first = apriori(df['3First'], min_support=0.075, min_confidence=0.2, min_lift=3, min_length=6)

for item in association_rules_first:
    pair = df['3First']
    items = [x for x in pair]
    print("Rule First " + items[0] + items[1] + items[2] + items[3] + items[4] + items[5] + items[6]) # + items[7] + items[8] + items[9] + items[10] + items[11] + items[12] + items[13] + items[14] + items[15] + items[16] + items[17] + items[18] ) #Item [2] és l'espai en blanc
    print("Support First: " + str(item[1]))
    print("Confidence First: " + str(item[2][0][2])) 
    print("Lift First: " + str(item[2][0][3]))
    print("=====================================")
