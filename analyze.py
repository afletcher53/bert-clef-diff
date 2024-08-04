import pandas as pd

results = pd.read_csv('results.csv')
average_percent_lost = results['percent_full'].mean()
average_words_lost = results['diff'].mean()
max_percent_lost = results['percent_full'].max()
max_words_lost = results['diff'].max()

print(f"Average percentage of words lost: {average_percent_lost:.2f}%")
print(f"Average number of words lost: {average_words_lost:.2f}")
print(f"Maximum percentage of words lost: {max_percent_lost:.2f}%")
print(f"Maximum number of words lost: {max_words_lost}")