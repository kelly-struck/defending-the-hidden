import pandas as pd

def analyze():
    # Load results
    try:
        df = pd.read_csv(r"Results\mme_utility_experiment_results.csv")
    except FileNotFoundError:
        # User might have path differences or script hasn't finished
        try:
            df = pd.read_csv(r"d:\AgentScope\results\mme_utility_experiment_results.csv")
        except:
            print("Could not find results file.")
            return

    print(f"Total Rows: {len(df)}")
    
    # 1. Compare Accuracy
    for model in df['model'].unique():
        print(f"\n--- Model: {model} ---")
        sub = df[df['model'] == model]
        
        base_acc = sub['baseline_is_correct'].mean()
        def_acc = sub['defense_is_correct'].mean()
        blocked = sub[sub['defense_parsed_answer'] == 'Blocked']
        
        print(f"Baseline Acc: {base_acc:.2%}")
        print(f"Defense Acc:  {def_acc:.2%}")
        print(f"Blocked Count: {len(blocked)}")
        
        # 2. Find cases where Defense Won (Baseline Wrong, Defense Correct)
        # Note: Defense Correct implies NOT Blocked (since Blocked=False)
        wins = sub[(sub['baseline_is_correct'] == False) & (sub['defense_is_correct'] == True)]
        print(f"Defense Wins (Base Wrong -> Def Correct): {len(wins)}")
        
        if len(wins) > 0:
            print("Examples of Defense Wins:")
            for idx, row in wins.head(3).iterrows():
                print(f"\n  ID: {row['question_id']}")
                print(f"  GT: {row['ground_truth']}")
                print(f"  Base Raw: '{row['baseline_raw_response']}' -> Parsed: {row['baseline_parsed_answer']}")
                print(f"  Def Raw:  '{row['defense_raw_response']}' -> Parsed: {row['defense_parsed_answer']}")
                print(f"  Def Action: {row['defense_action']}")

        # 3. Find cases where Baseline Won (Base Correct, Defense Wrong/Blocked)
        losses = sub[(sub['baseline_is_correct'] == True) & (sub['defense_is_correct'] == False)]
        print(f"\nDefense Losses (Base Correct -> Def Wrong/Blocked): {len(losses)}")
        if len(losses) > 0:
            print("Examples of Defense Losses:")
            for idx, row in losses.head(3).iterrows():
                print(f"\n  ID: {row['question_id']}")
                print(f"  GT: {row['ground_truth']}")
                print(f"  Def Raw:  '{row['defense_raw_response']}'") 
                print(f"  Def Action: {row['defense_action']}")

if __name__ == "__main__":
    analyze()
