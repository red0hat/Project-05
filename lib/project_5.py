from sklearn.model_selection import train_test_split
import sqlalchemy
import pandas as pd


def load_data_from_database(user_name=None, password=None, url=None, port='5432', database=None, table=None, **kwargs):
        """ This function opens a postgres database connection and loads a table into a pandas Dataframe.
             The default parameters are to connect to the dsi database hosted at joshuacook.me and 
             retrieve the madelon table.  """
        
### TO DO: Break out the check section to a separate function. 
###        and break the connection funtion in separate call, maybe.
        if user_name is None:
            user_name =  "dsi_student"
        if password is None:
            password =  "correct horse battery staple"
        if url is None:
            url =  "joshuacook.me"
        if database is None:
            database = 'dsi'
        if table is None:
            table = "madelon"

        engine = sqlalchemy.create_engine("postgresql://{}:{}@{}:{}/{}".format(user_name, password, url, port, database))
        df = pd.read_sql_table(table, engine)
        engine.dispose()
        return df

def make_data_dict(X, y, X_test=None, y_test=None, random_state=None, train_size = None):
    """
    This function instantiates a data dictionary. The format of the dictionary is a list of dictionaries.  
    Each time an operation is performed, a new dictionary with the model and the data should be 
    appended to the end of the list.
    
    If there a test set is not passed to the function, the X and y will be split into train adn test sets.
    
    """
   
### TO DO: consider not making it a list, if too much memory is tied up storing every step.
 
    if X_test is None and y_test is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=train_size)
    elif X_test is None or y_test is None:
        raise ValueError('You need to pass both X_test AND y_test or niether.')
    else:
        X_train = X
        y_train = y

    
    
    return [{'model' : '',
            'X_train' : X_train , 
            'y_train': y_train, 
            'X_test' : X_test, 
            'y_test' : y_test}]


def general_transformer(transformer, X_train, y_train, X_test, y_test):
    """
    This function transforms the data dictionary using sklearn, then returns a dictionaty of 
    the model and the modified the datasets.  The "X" data sets will be modified, based on the fit
    of the X_train dataset.  The y datasets will not be modified.
    """
    
    transformer.fit(X_train, y_train)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    
    return {'model': transformer,
            'X_train' : X_train,
            'y_train' : y_train,
            'X_test' : X_test,
            'y_test' : y_test}

            
            
def general_model(model, X_train, y_train, X_test, y_test):
    """
    This function applies a model from sklearn to the training set, then returns a dictionaty of the model,
    the datasets, and the scores for the model on the training and test sets.
    """
    
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return {'model': model,
            'X_train' : X_train,
            'y_train' : y_train,
            'X_test' : X_test,
            'y_test' : y_test,
            'train_score' : train_score,
            'test_score' : test_score}

