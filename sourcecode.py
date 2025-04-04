from customtkinter import *
import customtkinter
import threading
import tkinter




class MyWindow:
    def __init__(self, masterx, mastery):


        customtkinter.set_appearance_mode("dark")
        customtkinter.set_default_color_theme("dark-blue")

        self.root = customtkinter.CTk()
        self.root.geometry(f"{masterx}x{mastery}")
        self.root.resizable(width=True, height=True)
        self.root.after(201, lambda :self.root.iconbitmap(r"regression.ico"))
        self.root.title("Transportation's linear regression")
        self.PredictThread = None 

        self.MainCan = tkinter.Canvas(self.root, bg="#1a1a1a", highlightthickness=0)
        
        self.frame = customtkinter.CTkFrame(self.MainCan)
        self.features_frame = customtkinter.CTkFrame(self.frame, width=700, height=500, border_width=5, border_color="#0f4761")
        self.prediction_frame = customtkinter.CTkFrame(self.frame, width=350, height=500, border_width=5, border_color="#dbd70a")
        self.training_frame = customtkinter.CTkFrame(self.frame, border_width=5, border_color="#529949")
        self.Title = customtkinter.CTkLabel(self.frame, text="Transportation's traffic flow prediction \n using linear regression", anchor="center") 
        self.featuretitle = customtkinter.CTkLabel(self.features_frame, text="Input Features") 
        self.predictiontitle = customtkinter.CTkLabel(self.prediction_frame, text="Model Prediction")
        self.trainingtitle = customtkinter.CTkLabel(self.training_frame, text="Train Model")
        self.Timeofdayfeature = customtkinter.CTkLabel(self.features_frame, text="Time of day \n (in minutes)") 
        self.entry1 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.intervalfeature = customtkinter.CTkLabel(self.features_frame, text="Interval \n (No. every 4mins)") 
        self.entry2 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.speedfeature = customtkinter.CTkLabel(self.features_frame, text="Speed \n (miles)") 
        self.entry3 = customtkinter.CTkEntry(self.features_frame, state="disabled")
        self.predictionbutton = customtkinter. CTkButton(self.features_frame, state="disabled", text="Predict", command=self.thread_predict, fg_color="#0f4761", font=("Arial (Body CS)", 38), border_width=2, border_color="#1a1a1a")
        self.trainingbutton = customtkinter. CTkButton(self.training_frame, state="disabled", text="Train", command=self.thread_train, fg_color="#529949", font=("Arial (Body CS)", 38), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        self.answer = customtkinter.CTkEntry(self.prediction_frame, state="disabled", fg_color="#8e8b06", border_color="#343638")
        self.checkbox = customtkinter.CTkCheckBox(self.frame, text = "Full screen", command=self.fullscreen)
        self.selectfilebutton=customtkinter.CTkButton(self.training_frame,text="Select dataset", command=self.selectdataset, fg_color="#529949", font=("Arial (Body CS)", 38), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        self.pathdataset=customtkinter.CTkEntry(self.training_frame, state="disabled", fg_color="#529949", border_width=2, border_color="#1a1a1a")
        self.Place(1680,780)
        self.MainCan.bind("<Configure>", self.OnResize)
        self.root.mainloop()

    def Place(self, masterx, mastery):
        self.MainCan.place(relx=0.5, rely=0.5, relheight=1, relwidth=1,anchor=CENTER)
        self.frame.place(relx=0.5, rely=0.5, anchor=CENTER,  relwidth=0.97619, relheight=0.9487179)
        self.features_frame.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER, relwidth=(700/1680), relheight=(500/780))
        self.prediction_frame.place(relx=0.85, rely=0.6, anchor=tkinter.CENTER, relwidth=(350/1680), relheight=(500/780))
        self.training_frame.place(relx=0.15, rely=0.6, anchor=tkinter.CENTER, relwidth=(350/1680), relheight=(500/780))

        self.Title.place(relx=0.5, rely=0.12, anchor=tkinter.CENTER)
        self.Title.configure(font=("Arial (Body CS)", (28*((masterx+mastery)/1680))))

        self.featuretitle.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)
        self.featuretitle.configure(font=("Arial (Body CS)", (20*((masterx+mastery)/1680))))

        self.predictiontitle.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)
        self.predictiontitle.configure(font=("Arial (Body CS)", (20*((masterx+mastery)/1680))))

        self.trainingtitle.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)
        self.trainingtitle.configure(font=("Arial (Body CS)", (20*((masterx+mastery)/1680))))

        self.Timeofdayfeature.place(relx=0.2, rely=0.25, anchor=tkinter.CENTER)
        self.Timeofdayfeature.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.entry1.place(relx=0.2, rely=0.55, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(250/780))
        self.entry1.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.intervalfeature.place(relx=0.5, rely=0.25, anchor=tkinter.CENTER)
        self.intervalfeature.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.entry2.place(relx=0.5, rely=0.55, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(250/780))
        self.entry2.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.speedfeature.place(relx=0.8, rely=0.25, anchor=tkinter.CENTER)
        self.speedfeature.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))
        
        self.entry3.place(relx=0.8, rely=0.55, anchor=tkinter.CENTER, relwidth=(300/1680), relheight=(250/780))
        self.entry3.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
        self.predictionbutton.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))), corner_radius=30)

        self.trainingbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(800/1680), relheight=(150/780))
        self.trainingbutton.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))), corner_radius=30)

        self.selectfilebutton.place(relx=0.5, rely=0.35, anchor=tkinter.CENTER, relwidth=(1500/1680), relheight=(150/780))
        self.selectfilebutton.configure(font=("Arial (Body CS)", (14*((masterx+mastery)/1680))), corner_radius=30)

        self.pathdataset.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER, relwidth=(1500/1680), relheight=(150/780))
        self.pathdataset.configure(font=("Arial (Body CS)", (15*((masterx+mastery)/1680))))

        self.answer.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER, relwidth=(1000/1680), relheight=(250/780))
        self.answer.configure(font=("Arial (Body CS)", (18*((masterx+mastery)/1680))))

        self.checkbox.place(relx=0.95, rely=0.96, anchor=CENTER)
        
        
    def selectdataset(self):
        global filename
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select csv file", filetypes=[("dataset files","*.csv"),])
        self.pathdataset.configure(state="normal")
        self.pathdataset.insert(0, filename)
        self.pathdataset.configure(state="disabled")
        self.trainingbutton.configure(state="normal")




    def thread_predict(self):
        
        if self.PredictThread == None:
            self.PredictThread = threading.Thread(target=self.predict)
            self.PredictThread.start()
            return

        if self.PredictThread.is_alive() == False:
            self.PredictThread = threading.Thread(target=self.predict)
            self.PredictThread.start()
            return
        
    def thread_train(self):
        
        if self.PredictThread == None:
            self.PredictThread = threading.Thread(target=self.train)
            self.PredictThread.start()
            return

        if self.PredictThread.is_alive() == False:
            self.PredictThread = threading.Thread(target=self.train)
            self.PredictThread.start()
            return
        
    def train(self):
        self.trainingbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(1000/1680), relheight=(150/780))
        self.trainingbutton.configure(text="Training", state="disabled")
        import pandas
        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import r2_score
        import numpy as np


        global r2, rmse, mae
        global target_test, target_pred, best_model

        dataset = pandas.read_csv(filename)
        feature_columns = ['Time Period Ending', 'Time Interval', 'Avg mph']
        features = dataset[feature_columns]
        targetvariable = dataset['Total Volume']
        features_train, features_test, target_train, target_test = train_test_split(
            features, targetvariable, test_size=0.2, random_state=42)


        pipeline = Pipeline([
            ('poly', PolynomialFeatures(include_bias=False)),
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        param_grid = {'poly__degree': [2, 3, 4, 5], 'ridge__alpha': [0.01, 0.1, 1, 10, 100]}


        best_score = float('-inf')
        best_params = None
        for degree in param_grid['poly__degree']:
            for alpha in param_grid['ridge__alpha']:
                pipeline.set_params(poly__degree=degree, ridge__alpha=alpha)
                pipeline.fit(features_train, target_train)
                score = pipeline.score(features_test, target_test)
                if score > best_score:
                    best_score = score
                    best_params = {'poly__degree': degree, 'ridge__alpha': alpha}
        pipeline.set_params(**best_params)
        pipeline.fit(features_train, target_train)
        best_model = pipeline


        target_pred = best_model.predict(features_test)
        rmse = np.sqrt(mean_squared_error(target_test, target_pred))
        mae = mean_absolute_error(target_test, target_pred)
        r2 = r2_score(target_test, target_pred)

        self.trainingbutton.destroy()

        self.resultgbutton = customtkinter. CTkButton(self.training_frame, text="Results", command=self.result, fg_color="#529949", font=("Arial (Body CS)", 28), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        self.graphbutton = customtkinter. CTkButton(self.training_frame, text="Graph", command=self.graph, fg_color="#529949", font=("Arial (Body CS)", 28), border_width=2, border_color="#1a1a1a", hover_color="#366530")
        
        self.resultgbutton.place(relx=0.26, rely=0.85, anchor=tkinter.CENTER, relwidth=(800/1680), relheight=(150/780))
        self.graphbutton.place(relx=0.74, rely=0.85, anchor=tkinter.CENTER, relwidth=(800/1680), relheight=(150/780))

        self.entry1.configure(state="normal")
        self.entry2.configure(state="normal")
        self.entry3.configure(state="normal")
        self.predictionbutton.configure(state="normal")


    def result(self):
        self.resultgbutton.configure(state="disabled")
        import numpy as np
        new_window = customtkinter.CTkToplevel(self.root, fg_color="#212121")
        new_window.title("Results")
        new_window.geometry("425x275")
        new_window.resizable(width=False, height=False)


        def close():
            new_window.destroy()
            self.resultgbutton.configure(state="normal")

 
        self.titlewindow = customtkinter.CTkLabel(new_window, text=f"Performance indicator metrics", anchor="center", font=("Arial (Body CS)", 20))
        self.titlewindow.pack(pady=10)

        self.r2 = customtkinter.CTkLabel(new_window, text=f"R² Score: {f'R² Score: {r2:.4f}'}", anchor="center",font=("Arial (Body CS)", 20))
        self.r2.pack(pady=10)
        
        self.rmse = customtkinter.CTkLabel(new_window, text=f"Root Mean Squared Error (RMSE): {rmse:.4f}", anchor="center", font=("Arial (Body CS)", 20))
        self.rmse.pack(pady=10)

        self.mae = customtkinter.CTkLabel(new_window, text=f"Mean Absolute Error (MAE): {mae:.4f}", anchor="center", font=("Arial (Body CS)", 20))
        self.mae.pack(pady=10)

        new_button = customtkinter.CTkButton(new_window, text="Close Window", command=close)
        new_button.pack(pady=10)
        
        def confirm():
            new_window.destroy()
            self.resultgbutton.configure(state="normal")

        new_window.protocol("WM_DELETE_WINDOW", confirm)




    def graph(self):
        self.graphbutton.configure(state="disabled")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.scatter(target_test, target_pred, alpha=0.7)
        plt.xlabel('Actual Traffic Flow')
        plt.ylabel('Predicted Traffic Flow')
        plt.title('Actual vs Predicted Traffic Flow')
        plt.plot([min(target_test), max(target_test)], [min(target_test), max(target_test)], color='red')
        plt.show()
        self.graphbutton.configure(state="normal")



    def predict(self):
        self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(700/1680), relheight=(150/780))
        self.predictionbutton.configure(text="predicting",state="disabled")
        import pandas as pd
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        import matplotlib.pyplot as plt

        time_period = self.entry1.get()
        time_interval = self.entry2.get()
        avg_mph = self.entry3.get()


        _error = False

        try:
            self.entry1.configure(fg_color="#ff0000")
            float(time_period)
            self.entry1.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry2.configure(fg_color="#ff0000")
            float(time_interval)
            self.entry2.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        try:
            self.entry3.configure(fg_color="#ff0000")  
            float(avg_mph)
            self.entry3.configure(fg_color="#343638")
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
        except:
            _error = True
        if _error:
            self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
            self.predictionbutton.configure(text="predict",state="normal")
            return
        
        

        
        input_data = np.array([[time_period, time_interval, avg_mph]])
        input_data_transformed = best_model.named_steps['scaler'].transform(
            best_model.named_steps['poly'].transform(input_data)
        )
        predicted_value = best_model.named_steps['ridge'].predict(input_data_transformed)[0]
        
        self.answer.configure(state="normal")
        self.answer.delete(0, 'end')
        self.answer.insert(0, predicted_value)
        self.answer.configure(state="disabled")
        self.predictionbutton.place(relx=0.5, rely=0.85, anchor=tkinter.CENTER, relwidth=(500/1680), relheight=(150/780))
        self.predictionbutton.configure(text="predict",state="normal")
        
    def fullscreen(self):
        if self.checkbox.get():
            self.root.attributes("-fullscreen", True)
        else:
            self.root.attributes("-fullscreen", False)

    def OnResize(self, event):
        self.Place(event.width, event.height)



MyWindow(1680, 780)