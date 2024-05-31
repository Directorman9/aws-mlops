import subprocess, sys, os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("psycopg2-binary")
install("optuna")

import optuna, logging, psycopg2, argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score,f1_score, precision_score, recall_score, confusion_matrix
import gzip, pickle, pickletools


def bulk_insert(payload:list)->None:
    '''
    Inserts new customer with respective risk prediction score to the database.
    Input:
      payload: List of tuples of format [()]
    Output:
      void
    '''
    insert_statement = """INSERT INTO dotdataresults ("Prediction score", assurifiedleaseid, assurifiedcustomerid) VALUES (%s, %s, %s)"""

    try:
       cursor.executemany(insert_statement, payload)
       connection.commit()
       print("Bulk insert successful")

    except (Exception, psycopg2.DatabaseError) as error:
       print("Error while inserting data", error)

    finally:
       if connection:
          cursor.close()
          connection.close()
          print("PostgreSQL connection is closed")


def bulk_update(payload:list)->None:
    '''
    Updates the Prediction score of each record in the database table.
    Input:
      payload: List of tuples of format [()]
    Output:
      void
    '''
    update_statement = """UPDATE dotdataresults SET "Prediction score" = %s WHERE assurifiedleaseid = %s AND assurifiedcustomerid = %s ;"""

    try:
       cursor.c(update_statement, payload)
       connection.commit()
       print("Bulk update successful")

    except (Exception, psycopg2.DatabaseError) as error:
       print("Error while updating data", error)

    finally:
       if connection:
          cursor.close()
          connection.close()
          print("PostgreSQL connection is closed")


def mlp_experiment(trial):
    '''
    This method defines a single optuna experiment to test the MLP model.
    '''
    max_iter = trial.suggest_int("mlp_max_iter", 100, 500)
    lr_init = trial.suggest_float('learning_rate_init', 1e-5, 1e-1)

    mlp_model = MLPClassifier(learning_rate='adaptive', max_iter=max_iter, learning_rate_init=lr_init, random_state=0)
    f1_scorer = make_scorer(f1_score, average='macro')

    score = cross_val_score(mlp_model, features, labels, n_jobs=-1, cv=3, scoring=f1_scorer).mean()

    return score


def rf_experiment(trial):
    '''
    This method defines a single optuna experiment to test the random forest model.
    '''
    rf_max_depth = trial.suggest_int("rf_max_depth", 2, 40, log=False)
    n_estimators = trial.suggest_int('n_estimators', 10, 100, log=False)

    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=rf_max_depth)
    f1_scorer = make_scorer(f1_score, average='macro')

    score = cross_val_score(rf_model, features, labels, n_jobs=-1, cv=3, scoring=f1_scorer).mean()

    return score



if __name__ == "__main__": 

   # Arguments
   parser = argparse.ArgumentParser()
   parser.add_argument('--customer_id', type=str)
   args = parser.parse_args()

   customer_id = args.customer_id
   output_dir =  "/opt/ml/processing/output"
   os.makedirs(output_dir, exist_ok=True)
   
   # Database connection details
   db_params ={
        'host': os.getenv("DB_HOST"),
        'port': os.getenv("DB_PORT"),
        'dbname': os.getenv("DB_NAME"),
        'user': os.getenv("DB_USER"),
        'password': os.getenv("DB_PASS"),
   }
  
   # Establish database connection
   connection = psycopg2.connect(**db_params)
   cursor = connection.cursor()


   # Pull customer data 
   query = f"SELECT * FROM dotdatatraining WHERE assurifiedcustomerid = '{customer_id}'"
   df = pd.read_sql(query, connection)


   #Preprocess
   features = df[["walkscore", "sqft", "start_month", "lease_duration", "rentamount", "total_unpaid_rent", "covid"]]
   labels = df['loss_indicator']
   train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, shuffle=True, random_state=0)

    
   #hyperparameter tuning
   optuna.logging.set_verbosity(optuna.logging.WARNING)

   mlp_study = optuna.create_study(direction="maximize")
   mlp_study.optimize(mlp_experiment, n_trials=100)
   print(mlp_study.best_trial)

   rf_study = optuna.create_study(direction="maximize")
   rf_study.optimize(rf_experiment, n_trials=100)
   print(rf_study.best_trial)

    
   #Train
   #MLP
   max_iter = mlp_study.best_trial.params.get("mlp_max_iter")
   lr_init = mlp_study.best_trial.params.get("learning_rate_init")
   mlp_model = MLPClassifier(learning_rate='adaptive', max_iter=max_iter, learning_rate_init=lr_init, random_state=0).fit(train_features,train_labels)

   #RF
   rf_max_depth = rf_study.best_trial.params.get("rf_max_depth")
   n_estimators = rf_study.best_trial.params.get("n_estimators")
   rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=rf_max_depth, random_state=0).fit(train_features,train_labels)


   #Evaluate
   mlp_predictions = mlp_model.predict(test_features)
   rf_predictions = rf_model.predict(test_features)


   #Reporting
   #MLP
   print ('accuracy is: ', accuracy_score(test_labels, mlp_predictions))
   print ('precision (B): ', precision_score(test_labels, mlp_predictions, average='binary'))
   print ('precision (M): ', precision_score(test_labels, mlp_predictions, average='macro'))
   print ('recall (B): ',  recall_score(test_labels, mlp_predictions, average='binary'))
   print ('recall (M): ',  recall_score(test_labels, mlp_predictions, average='macro'))
   print ('F1 (B): ', f1_score(test_labels, mlp_predictions, average='binary'))
   mlp_f1_score = f1_score(test_labels, mlp_predictions, average='macro')
   print ('F1 (M) ', mlp_f1_score)
   print ('Confusion_matrix is: ',  confusion_matrix(test_labels, mlp_predictions, labels=[1,0]))


   #RF
   print ('accuracy is: ', accuracy_score(test_labels, rf_predictions))   
   print ('precision (B): ', precision_score(test_labels, rf_predictions, average='binary'))
   print ('precision (M): ', precision_score(test_labels, rf_predictions, average='macro'))
   print ('recall (B): ',  recall_score(test_labels, rf_predictions, average='binary'))
   print ('recall (M): ',  recall_score(test_labels, rf_predictions, average='macro'))
   print ('F1 (B): ', f1_score(test_labels, rf_predictions, average='binary'))
   rf_f1_score = f1_score(test_labels, rf_predictions, average='macro')
   print ('F1 (M) ', rf_f1_score)
   print ('Confusion_matrix is: ',  confusion_matrix(test_labels, rf_predictions, labels=[1,0]))


   #Save best model
   if mlp_f1_score > rf_f1_score:
      best_model = mlp_model
   else:
      best_model = rf_model

   with gzip.open(f"{output_dir}/best_model.p", "wb") as f:
       pickled = pickle.dumps(best_model)
       optimized_pickle = pickletools.optimize(pickled)
       f.write(optimized_pickle)


   #Predict on new customer
   query = f"SELECT * FROM dotdatalosstotalpredictiondata WHERE assurifiedcustomerid = '{customer_id}'"
   df = pd.read_sql(query, connection)

   features = df[["walkscore", "sqft", "start_month", "lease_duration", "rentamount", "total_unpaid_rent", "covid"]]

   classes_proba = best_model.predict_proba(features)
   df['prediction_score'] = classes_proba[:,1]

   payload = df[['prediction_score', 'assurifiedleaseid', 'assurifiedcustomerid']]
   #payload = payload.loc[payload["prediction_score"] > 0.83]
   payload = payload.values.tolist()

   bulk_insert(payload)
   print ("Process completed.")
