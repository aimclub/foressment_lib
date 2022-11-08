import pandas as pd

if __name__ == '__main__':
    
    import sys
    
    if sys.argv:
        
        test_path = sys.argv[0]
        class_column = sys.argv[1]
        positive_class_label = sys.argv[2]

        test_data = pd.read_csv(test_path)

        algo = IzdapAlgo(0.1)
        algo.fit(test_data, class_column = class_column, positive_class_label = positive_class_label)

        rules = algo.get_rules()
        print(rules[0])

        transformed_data = algo.transform(test_data)
        print(transformed_data.info())    