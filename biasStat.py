import pandas as pd
import numpy as np
from scipy import stats

# Hardcoded values
data = [
    {
        'Name': "example",
        'Male_correct': 10,
        'Female_correct': 2,
        'Male_incorrect': 9,
        'Female_incorrect': 1
    },
    {
        'Name': "GAN-REAL CNN",
        'Male_correct': 410 + 447,
        'Male_incorrect': 26 + 2,
        'Female_correct': 545 + 551,
        'Female_incorrect': 19 + 0
    },
    {
        'Name': "GAN-REAL TL",
        'Male_correct': 361 + 449,
        'Male_incorrect': 75 + 0,
        'Female_correct': 480 + 462,
        'Female_incorrect': 84 + 89
    },
    {
        'Name': "SD-REAL CNN",
        'Male_correct': 1974 + 449,
        'Male_incorrect': 26 + 0,
        'Female_correct': 1992 + 551,
        'Female_incorrect': 8 + 0
    },
    {
        'Name': "SD-REAL TL",
        'Male_correct': 1848 + 449,
        'Male_incorrect': 152 + 0,
        'Female_correct': 1994 + 551,
        'Female_incorrect': 6 + 0
    },
]

def get_stats(correct_male, incorrect_male, correct_female, incorrect_female):
    # Total counts
    total_male = correct_male + incorrect_male
    total_female = correct_female + incorrect_female
    
    # Proportions
    prop_correct_male = correct_male / total_male
    prop_correct_female = correct_female / total_female
    prop_incorrect_male = incorrect_male / total_male
    prop_incorrect_female = incorrect_female / total_female
    
    # Statistical tests
    pooled_p_correct = (correct_male + correct_female) / (total_male + total_female)
    pooled_p_incorrect = (incorrect_male + incorrect_female) / (total_male + total_female)
    
    test_stat_correct = (prop_correct_male - prop_correct_female) / np.sqrt(pooled_p_correct * (1 - pooled_p_correct) * (1 / total_male + 1 / total_female))
    test_stat_incorrect = (prop_incorrect_male - prop_incorrect_female) / np.sqrt(pooled_p_incorrect * (1 - pooled_p_incorrect) * (1 / total_male + 1 / total_female))
    
    p_value_correct = 2 * (1 - stats.norm.cdf(np.abs(test_stat_correct)))
    p_value_incorrect = 2 * (1 - stats.norm.cdf(np.abs(test_stat_incorrect)))
    
    results = {
        'proportion_correct_male': prop_correct_male,
        'proportion_correct_female': prop_correct_female,
        'proportion_incorrect_male': prop_incorrect_male,
        'proportion_incorrect_female': prop_incorrect_female,
        'test_statistic_correct': test_stat_correct,
        'p_value_correct': p_value_correct,
        'test_statistic_incorrect': test_stat_incorrect,
        'p_value_incorrect': p_value_incorrect
    }
    
    return results

def main():
    
    for idx in range(len(data)):
        print(f"Results for {data[idx]['Name']}")
        correct_male = data[idx]['Male_correct']
        incorrect_male = data[idx]['Male_incorrect']
        correct_female = data[idx]['Female_correct']
        incorrect_female = data[idx]['Female_incorrect']
        
        results = get_stats(correct_male, incorrect_male, correct_female, incorrect_female)
        
        # Convert results to DataFrame for easy saving
        df_results = pd.DataFrame([results])
        df_results.to_csv("fairness_metrics.csv", index=False)
        
        # Print results
        print("Proportion of correct predictions for males: {:.6f}".format(results['proportion_correct_male']))
        print("Proportion of correct predictions for females: {:.6f}".format(results['proportion_correct_female']))
        print("Proportion of incorrect predictions for males: {:.6f}".format(results['proportion_incorrect_male']))
        print("Proportion of incorrect predictions for females: {:.6f}".format(results['proportion_incorrect_female']))
        print("Test statistic for correct predictions: {:.6f}".format(results['test_statistic_correct']))
        print("P-value for correct predictions: {:.6f}".format(results['p_value_correct']))
        print("Test statistic for incorrect predictions: {:.6f}".format(results['test_statistic_incorrect']))
        print("P-value for incorrect predictions: {:.6f}".format(results['p_value_incorrect']))
           
        # Determine which gender is more accurate (if p-value < 0.05)
        if results['p_value_correct'] < 0.05:
            if results['proportion_correct_male'] > results['proportion_correct_female']:
                print("Male gender is significantly more accurate.")
            else:
                print("Female gender is significantly more accurate.")
        else:
            print("Neither gender is significantly more accurate.")
        print()

if __name__ == "__main__":
    main()
