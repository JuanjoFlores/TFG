import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import sklearn 
from math import sqrt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from pytorch_lightning.metrics import ConfusionMatrix
from dataset import myDataset
from model import Net

class main():
    def trainEvaluation(all,learning_rate,me,nf):

        torch.manual_seed(1)
        np.random.seed(1)
        max_epochs = me
        num_feats = nf #1024
        batch_size = 64
        n_classes = 9
        layers = [256]#[0]
        dropout = 0.5
        gpu = 0
        use_gpu = True
        device = torch.device('cuda:{}'.format(gpu) if use_gpu else 'cpu')
        #Train Loading
        data_tr, len_feats, n_classes = myDataset.load('./vector_tfidf_train', num_feats)
        
        dataset_tr = myDataset(data_tr)
        data_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size,shuffle=True, num_workers=0)
        
        #Test Loading 
        data_te,_,_ = myDataset.load('./vector_tfidf_test', num_feats)

        dataset_te = myDataset(data_te)

        data_te = torch.utils.data.DataLoader(dataset_te, batch_size=batch_size,shuffle=False, num_workers=0)
        
        model = Net(len_feats, layers, n_classes = n_classes,dropout=dropout) #CREAMOS EL MODELO
        
        if(all==1):
            print(model)
            print(f"Número de parámetros:{model.num_params}")

        model.to(device)
        criterion = torch.nn.NLLLoss() #Loss function Cross Entropy 
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) #Optimizer -> Stochastic Gradient Descendent with learning rate = 0.01
        
        best_train_loss = 99999
        best_epoch = 0
        best_model = None
        for epoch in range(max_epochs):
            train_loss = 0.0
            contador = 1 
            model.train()#Model in train mode
            for batch, sample in enumerate(data_tr):
            
                c = sample['row'].to(device) #Obtenemos las caracteristicas y las pasamos a device
                l = sample['label'].to(device) #Obtenemos la etiqueta de clase y la pasamos a device
                
                optimizer.zero_grad()
                
                outputs = model(c) #SALIDA DE FORWARD -> USAR LUEGO EN EVALUACIÓN
                #Compute Loss
                loss = criterion(outputs, l)
                
                loss.backward() #Aplicamos el algoritmo de backpropagation
                optimizer.step()
                
                train_loss += loss.item() / l.data.size()[0]
                contador += 1
                
            train_loss = train_loss / contador #Normalizamos
            if(all==1):
                print('Epoch {}: train loss: {}'.format(epoch, train_loss))

            if train_loss <= best_train_loss: 
                best_train_loss = train_loss
                best_epoch = epoch
                best_model = model.state_dict()

        if(all==1):
            print('Best Epoch {}: Best train loss: {}'.format(best_epoch, best_train_loss))    
            print('-'*50)    
        
        #Cargamos best_model
        #print(f'Loading Best model, from epoch:{best_epoch}')
        model.load_state_dict(best_model)

        #EVALUATION
        model.eval() #Model in evaluation mode
        test_loss = 0.0
        class_correct = list(0. for i in range(n_classes)) #Clases que el sistema ha acertado
        class_total = list(0. for i in range(n_classes)) #Todas las clases
        correct_array = np.zeros((n_classes,n_classes))
        hipotesis_array = np.zeros((n_classes,n_classes))
        contador = 0
        confmat = np.zeros((n_classes,n_classes))
        for batch, sample in enumerate(data_te):

            c = sample['row'].to(device)
            l = sample['label'].to(device)

            output = model(c)
            loss = criterion(output,l)

            test_loss += loss.item()*c.size(0)

            _, pred = torch.max(output, 1)
            correct = pred.eq(l) #Comprobamos si el objeto pertence a la clase o no.
            correct_array =  l
            hipotesis_array = pred
            #Sklearn method:
            #hipotesis_array.extend(pred.cpu().detach().numpy())
            #correct_array.extend(l.cpu().detach().numpy())
            for i in range(c.size(0)):  
                label = l.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            contador += 1
            #Compute of Confusion Matrix
            cm = confusion_matrix(correct_array.cpu(),hipotesis_array.cpu(),labels=range(n_classes))
            confmat = confmat + cm

        
        
        #resSK = accuracy_score(correct_array,hipotesis_array)
        #print(resSK)
        test_loss = test_loss / contador #Normalizamos
        if(all==1):
            print('Test loss: {}'.format(test_loss))

        #Accuracy for each class
        if(all==1):
            for i in range(n_classes):
                if class_total[i] > 0:
                    print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (str(i+1), 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])))
                else:
                    print('Test Accuracy of %5s: N/A (no training examples)' % (class_total[i]))
        
        #Total accuracy
        res = (100. * np.sum(class_correct) / np.sum(class_total))#,np.sum(class_correct), np.sum(class_total))
        #print('\nTest Accuracy (Overall): %2d%%' % res)

        #Confusion Matrix 
        if(all==1):
            print("Confusion Matrix (Overall):",confmat)

        # Confidence interval:
        interval = (1.96 * sqrt((res/100*(1-res/100))/np.sum(class_total)))*100
        #print(f'Confidence Interval Radius:{interval}')

        return res,interval,confmat
