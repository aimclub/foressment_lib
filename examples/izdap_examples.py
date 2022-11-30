import pandas as pd
import sys
from aopssop import IzdapAlgo

if __name__ == '__main__':
    
    if sys.argv:
        
        data = make_classification(n_samples=200, n_features=4, 
                                   n_informative=2, n_classes=2,
                                   random_state=42)
        
        class_column = 'class'
        positive_class_label = 1
        probability_threshold = 0.1
        
        test_data = pd.DataFrame(data[0], columns = ['col1','col2','col3','col4'])
        test_data[class_column] = pd.Series(data[1])

        algo = IzdapAlgo(probability_threshold)
        algo.fit(test_data, class_column=class_column, positive_class_label=positive_class_label)

        rules = algo.get_rules()
        print(rules[0])

        transformed_data = algo.transform(test_data)
        print(transformed_data.info())    
