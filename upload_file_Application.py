import pandas as pd
import openpyxl
from openpyxl import load_workbook
import numpy as np
import time
from tkinter import *
from tkinter import ttk
from pandastable import Table, TableModel
import os 
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
from src.models.ClearTransformData import Cleardataset
from src.models.Word2VecTransformer import *
from src.models.FastTextTransformer import FastTextTransformer
from tkinter import messagebox
from src.models.ClearTransformData import parallelize_dataframe
 
global dataframe
global dataFr
global combo_box 
global combo_boxVille 
global combo_Town
combo_box=None
combo_boxVille=None

class UserInterface(Frame):

    
    # Launch the df in a pandastable frame
    def __init__(self, parent=None):
        Frame.__init__(self)
        global ui_df
        global dataFr
        ui_df = dataFr=dataframe
        self.parent = parent
        self.refresh_df(df = ui_df)


    def refresh_df(self, df,boolTest=False):
        combo_selectionLabel = None
        combo_selectionVille = None
        if bool(combo_box) and bool(combo_boxVille) and boolTest:
            combo_box.set("")
            combo_boxVille.set("")
        Frame.__init__(self)
        self.main = self.master
        f = Frame(self.main)
        f.grid(row =1, column = 0, rowspan = 6, columnspan = 3, sticky = W+E+N+S)
        screen_width = f.winfo_screenwidth() * 0.9
        screen_height = f.winfo_screenheight() * 0.5
        print(df)
        dataFr= df
        self.table = pt = Table(f, dataframe=df, height = screen_height, width = screen_width,showtoolbar=True, showstatusbar=True)
        pt.show()
        return


     
    def change_df_combo(self,event):
        #Responds to combobox, supposed to filter by 'Sec_type'
        combo_selectionLabel = str(combo_box.get())
        combo_selectionVille = str(combo_boxVille.get())
        print(combo_selectionLabel)
        print(combo_selectionVille)
        if  bool(combo_selectionLabel) and bool(combo_selectionVille):
            ui_df=dataframe[(dataframe.Labels == combo_selectionLabel) & (dataframe.ville==combo_selectionVille)]
        elif not  bool(combo_selectionLabel) and  bool(combo_selectionVille):
            ui_df = dataframe[dataframe["ville"]==combo_selectionVille]
        elif bool(combo_selectionLabel) and not bool(combo_selectionVille):
            ui_df = dataframe[dataframe["Labels"] == combo_selectionLabel]

        print("None error")
        print("shape:",ui_df.shape)
        self.refresh_df(df=ui_df)

    def change_df_combo_job(self,event):
        combo_selectionLabel = str(combo_box.get())
        ui_df = dataframe[dataframe["Labels"] == combo_selectionLabel]
        self.refresh_df(df=ui_df)
        
    def change_df_combo_Ville(self,event):
        combo_selectionVille = str(combo_boxVille.get())
        ui_df = dataframe[dataframe["ville"]==combo_selectionVille]
        self.refresh_df(df=ui_df)

    
class UploadData(Frame):

    def __init__(self,master):
        Frame.__init__(self, master)
       

        self.call_button=Button(text ='Upload data',command =self.import_csv_data)
        self.call_button.grid(column=0,row =3, sticky = (N, W, E, S))
        self.quit_button = Button(text ='Quitt',command =self.quit)
        self.quit_button.grid(column=3,row =3, sticky = (N, W, E, S))

        self.grid()
         
    def import_csv_data(self):
        global dataframe
        csv_file_path = askopenfilename()
        
        root2= self.master
        
        self.call_button.destroy()
        self.quit_button.destroy()
        
        bestModel = pickle.load(open("src/data/model_with_resum/model_with_resummodel_resume_tfidf_clfLogistic.sav","rb"))

        dataframe = pd.read_csv(csv_file_path)
        dataframe.insert(loc=1,column ="idJob",value=["id_"+str(i) for i in dataframe.index] )
        columnName =  dataframe.columns
        while "ville" not in columnName or "resume" not in columnName:
            messagebox.showinfo("Error!!!", "Please, enter the right type of datafrme!")

            self.call_button.destroy()
            self.quit_button.destroy()

            self.call_button=Button(text ='Upload data',command =self.import_csv_data)
            self.call_button.grid(column=0,row =3, sticky = (N, W, E, S))
            self.quit_button = Button(text ='Quitt',command =self.quit)
            self.quit_button.grid(column=3,row =3, sticky = (N, W, E, S))
            self.grid()
            dataframe = pd.read_csv(csv_file_path)
            dataframe.insert(loc=1,column ="idJob",value=["id_"+str(i) for i in dataframe.index] )
            columnName =  dataframe.columns

        self.call_button.destroy()
        self.quit_button.destroy()
        subWindow = Toplevel(root2)
        subWindow.title("Please Wait,we are processing data and Predict Job")
        subWindow.geometry("450x40")
        progress = ttk.Progressbar(subWindow, orient = HORIZONTAL, length = 420, mode = 'determinate')
        progress.grid(column=0, row=2)

        progress.step(0)
        progress.update()
    
        progress.step(10)
        subWindow.update()
        classClear =  Cleardataset()
        clearResumedata = parallelize_dataframe(dataframe.resume,classClear.transform)
        progress.step(60)
        subWindow.update() 
        Labels = bestModel.predict(clearResumedata)
        dataframe.insert(loc=0, column='Labels', value=Labels)
        progress.step(100)
        subWindow.update() 
        progress.destroy()
        subWindow.destroy()
        
        app2 = ApplicationJobPredict(root2)
        
### Lauch Tkinter


class PlotGraph:
    
    def __init__(self,rootFrame):
       
        self.master = rootFrame
        global TownSelected
        #self.relauchPlot(TownSelected)
        
        
    def __plot_diag(self,x,x_label):
    

        plt.style.use('classic')
        fig=plt.figure(figsize=(12,9))
        ax = plt.axes(facecolor='#E6E6E6')

        # Afficher les ticks en dessous de l'axe
        ax.set_axisbelow(True)

        # Cadre en blanc
        plt.grid(color='w', linestyle='solid')

        # Cacher le cadre
        # ax.spines contient les lignes qui entourent la zone où les 
        # données sont affichées.
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Cacher les marqueurs en haut et à droite
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

        # Nous pouvons personnaliser les étiquettes des marqueurs
        # et leur appliquer une rotation
        marqueurs = np.arange(len(x))
        xtick_labels = x_label
        ax.xaxis.set_tick_params(labelsize=9)
        plt.xticks(marqueurs, xtick_labels, rotation=25)

        # Changer les couleur des marqueurs
        ax.tick_params(colors='gray', direction='out')
        for tick in ax.get_xticklabels():
            tick.set_color('gray')
        for tick in ax.get_yticklabels():
            tick.set_color('gray')
        
        # Changer les couleur des barres
        ax.bar(marqueurs,x, edgecolor='#E6E6E6', color='#EE6666');

        return fig

             
    def create_plot(self, TownSelected):
        
        res = dataframe[dataframe.ville==TownSelected].loc[:,["ville","Labels"]].groupby("Labels").count()
    
        valeur = res.ville.tolist()
        x_label= (x for x in res.index.tolist())
        x = np.arange(len(valeur))


        fig = self.__plot_diag(valeur,x_label)
        canvas = FigureCanvasTkAgg(fig, master=self.master)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=1, columnspan=2, rowspan=2, padx=5, pady=5)
    
        
    def changeGraph(self,event):
        TownSelected = str(combo_Town.get())
        self.create_plot(TownSelected)



## Main function:


def ApplicationJobPredict(master):

    global combo_box
    global combo_boxVille

    
    mainframe = ttk.Frame(master,padding="32 32 42 42")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)
    master.columnconfigure(0, weight=1)
    master.rowconfigure(0, weight=1)

    
    
    ui_display = UserInterface(mainframe)

    ## Function use to plot diagrame in a button

    def run():
        global combo_Town
        rootFrame = Toplevel()
        plotbarPlot = PlotGraph(rootFrame)
        plotbarPlot.create_plot("Chicago")
        #create_plot(rootFrame,TownSelected="Chicago")
        combo_choicesTown = list(np.unique(dataframe.ville))
        choiceTown = StringVar()
        combo_Town = ttk.Combobox(rootFrame, textvariable=choiceTown)
        combo_Town['values'] = combo_choicesTown
        labelTown = ttk.Label(rootFrame,text = "Choise City")
        labelTown.grid(column=1, row=0)
        combo_Town.grid(column=1, row=1)
        TownSelected = str(combo_Town.get())
        combo_Town.bind("<<ComboboxSelected>>",plotbarPlot.changeGraph)

    
    buttomPlot = Button(mainframe,text ="View partition in a city",command = run)
    buttomPlot.grid(column=2,row=2)

    def runShowDescription():

        global currentIndex 
        
        frameRoot = Toplevel()
        frameRoot.title("show offer description")
        indexChoices = list(dataFr.idJob)
        choiceIndex = StringVar()
        indexBox = ttk.Combobox(frameRoot,textvariable = choiceIndex)
        indexBox["values"] = indexChoices
        indexBox.grid(column=0,row=0)
       
        
        def refresh_description(event):
            currentIndex = str(indexBox.get())
            text2 = Text(frameRoot, height=40, width=120)
            scroll = Scrollbar(frameRoot, command=text2.yview)
            text2.configure(yscrollcommand=scroll.set)
            text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
            text2.tag_configure('big', font=('Verdana', 20, 'bold'))
            text2.tag_configure('color',
                                foreground='#476042',
                                font=('Tempus Sans ITC', 8, 'bold'))
            text2.tag_bind('follow',
                           '<1>',
                           lambda e, t=text2: t.insert(END, "Not now, maybe later!"))
            text2.insert(END,'\n Offer description\n', 'big')
            ind = "id_"+str(1)
            print("current index={}".format(currentIndex))
            if currentIndex != '':
                print("-------------------change index-----------------")
                ind = currentIndex
            quote =  list(dataFr.description.loc[dataFr.idJob==ind])[0]
            text2.insert(END, quote, 'color')
            text2.grid(row = 1,column=0)
            scroll.grid(row=1,column=1,padx=12,sticky='NS')

        refresh_description(event=None)

        indexBox.bind("<<ComboboxSelected>>",refresh_description)
        

        


    buttondescription = Button(mainframe,text="View description",command = runShowDescription)
    buttondescription.grid(column = 1, row = 2)
    
    test_button = Button(mainframe, text = 'Relancer dataframe', command= lambda: ui_display.refresh_df(dataframe,boolTest=True))
    test_button.grid(column=0, row=2)


    combo_choices = list(np.unique(dataframe.Labels))
    choice = StringVar()
    combo_box = ttk.Combobox(mainframe, textvariable=choice)
    #combo_box = Co(master=mainframe,list_of_items=choice,highlightthickness=1)
    combo_box['values'] = combo_choices
    labelTop = ttk.Label(mainframe,text = "Job list")
    labelTop.grid(column=2, row=0)
    combo_box.grid(column=2, row=1)
    combo_box.bind("<<ComboboxSelected>>",ui_display.change_df_combo)


    combo_choicesVille = list(np.unique(dataframe.ville))
    choiceVille = StringVar()
    combo_boxVille = ttk.Combobox(mainframe, textvariable=choiceVille)
    combo_boxVille['values'] = combo_choicesVille
    labelTopVille = ttk.Label(mainframe,text = "City")
    labelTopVille.grid(column=1, row=0)
    combo_boxVille.grid(column=1, row=1)

    combo_boxVille.bind("<<ComboboxSelected>>",ui_display.change_df_combo)

   

if __name__=="__main__":
    
        
    root =Tk()
    root.title("Tools for job prediction")
    root.geometry("800x600")
   
    app = UploadData(root)
    root.mainloop()

