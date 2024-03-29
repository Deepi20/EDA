def replace():
    print()
    print(format('How to replace multiple values in a Pandas DataFrame','*^82'))
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import numpy as np
    raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
                'last_name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze'],
                'age': [42, 52, 36, 24, 73],
                'preTestScore': [-999, -999, -999, 2, 1],
                'postTestScore': [2, 2, -999, 2, -999]}
    df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])
    print(); print(df)
    print(); print(df.replace(-999, np.nan))
    print(); print(df.replace(-999, 0))
    
    replace()
