
#Association Rule Learning is an approach that discovers the strenght of 
#relationships between different data points
#it is commonly used to understand which products are purchased together
#Association Rule learning

#Apriori
#metrics:
#support-% of transaction that contain item A and B, both transactions that contain item A and B/ number of transactions
#confidence -relationship between 2 items, of all transactions that included item A what proportion also included tem B
#expected confidence- % number of transactions that container item B
#lift- the factor by which the confidence exceeds the expected confidence

#########################
#Association rule learning using apriori
#########################



#########################
#import required packages
########################

from apyori import apriori
import pandas as pd




#######################
#import data
######################



#import


alcohol_transactions = pd.read_csv("data/sample_data_apriori.csv")



#drop ID column

alcohol_transactions.drop("transaction_id", axis = 1, inplace = True)

#modify data for apriori algorithm

transactions_list =  []
for index, row in alcohol_transactions.iterrows():
    transaction = list(row.dropna())
    transactions_list.append(transaction)
 

#apply the apriori algorithm

apriori_rules = apriori (transactions_list,
                         min_support = 0.003,
                         min_confidence = 0.2,
                         min_lift = 3,
                         mi_lenght = 2,
                         max_lenght = 2)

apriori_rules = list(apriori_rules)

apriori_rules =[0]

#convert output to dataframe

product1 = [list(rule[2][0][0])[0] for rule in apriori_rules]
product2 = [list(rule[2][0][1])[0] for rule in apriori_rules]
support = [rule[1] for rule in apriori_rules]
confidence = [rule[2[0][2]] for rule in apriori_rules]
lift = [rule[2][0][3] for rule in apriori_rules]

apriori_rules_df = pd.DataFrame({"product1":product1,                    
                                "product2" : product2,
                                 "support": support,
                                 "confidence" : confidence,
                                 "lift" : lift})
##################################
#sort rules by descending lift
#################################

apriori_rules_df.sort_values(by = "lift", ascending = False, inplcace = True)

############################################
#search rules
###########################################

apriori_rules_df[apriori_rules_df["product1"].str.contains("New Zealand")]