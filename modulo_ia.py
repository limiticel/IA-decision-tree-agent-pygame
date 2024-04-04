# Importação de bibliotecas necessárias
from sklearn.tree import DecisionTreeClassifier  # Importa o classificador de árvore de decisão do scikit-learn
from sklearn.metrics import accuracy_score  # Importa a função para calcular a acurácia do modelo
import random  # Importa a biblioteca random para geração de números aleatórios
import pygame  # Importa a biblioteca pygame para criação da interface gráfica
import math  # Importa a biblioteca math para cálculos matemáticos
import sys  # Importa a biblioteca sys para manipulação do sistema
from time import sleep  # Importa a função sleep para pausas no programa
import numpy as np  # Importa a biblioteca numpy para operações numéricas

# Definição da classe IA_class
class IA_class:
    def __init__(self, state, memory, distance_goal, act, epslon=0.1, alpha=0.03, model=DecisionTreeClassifier()):
        """
        Inicializa a classe IA_class.

        Args:
            state (list): Estado inicial do agente.
            memory (int): Memória do agente (ainda não utilizada).
            distance_goal (float): Distância até o objetivo.
            act (int): Ação inicial do agente.
            epslon (float): Epslon para decisões aleatórias (padrão: 0.1).
            alpha (float): Taxa de aprendizado (padrão: 0.03).
            model (DecisionTreeClassifier): Modelo de classificação (padrão: DecisionTreeClassifier).
        """
        self.memory = memory
        self.memory_ = []  # A memória do agente (ainda não implementada)
        self.template = []  # Template para treinamento do modelo
        self.model = model  # Modelo de classificação
        self.definition = 0  # Definição (ainda não utilizada)
        self.state = list(state)  # Estado do agente (posição)
        self.table_t = []  # Tabela de treinamento do modelo
        self.color = (255, 0, 0)  # Cor do agente
        self.especifier_ia = [20, 20]  # Especificação do agente
        self.SPEED = 5  # Velocidade do agente
        self.vision = 350  # Visão do agente
        self.distance_goal = distance_goal  # Distância até o objetivo
        self.visible_ = False  # Indica se o objetivo está visível
        self.rate = -0.1  # Taxa de classificação (ainda não utilizada)
        self.goal_pos = 0  # Posição do objetivo
        self.epslon = epslon  # Epslon para decisões aleatórias
        self.punt_sec = [0, 0]  # Pontuação do agente
        self.limit = 0  # Limite de alerta

    # Métodos para manipulação do ambiente e interface gráfica
    def set_memory_(self):
        """Função para definir a memória do agente (ainda não implementada)."""
        pass

    def indexing(self, screen, floor):
        """
        Desenha o agente na tela.

        Args:
            screen: Superfície da tela onde o agente será desenhado.
            floor (int): Altura do chão na tela.
        
        Returns:
            pygame.Rect: Retângulo representando o agente.
        """
        return pygame.draw.rect(screen, self.color, (self.state[0], floor - self.especifier_ia[0], self.especifier_ia[0], self.especifier_ia[1]))

    def sensor_(self, num_sensor, w_screen, h_screen, screen):
        """
        Gera sensores à direita do agente.

        Args:
            num_sensor (int): Número de sensores.
            w_screen (int): Largura da tela.
            h_screen (int): Altura da tela.
            screen: Superfície da tela.
        
        Returns:
            float: Distância até o objetivo.
        """
        # Cálculos para geração dos sensores à direita do agente
        angle_init=math.pi/0.5
        angle_inc=math.pi*1 /(num_sensor -1)

        com=random.uniform(30,self.vision) #largura do sensor por frame

        for i in range(0,num_sensor):
            
            angle= angle_init + i * angle_inc

            sensor_x=int(self.state[0]+ 20 // 2)
            sensor_y=int(self.state[1]+ 20 // 2)

            end_x = int(sensor_x + com * math.cos(angle)) 
            end_y = int(sensor_y + com * math.sin(angle)) 

            if 0<= end_x < w_screen and 0 <= end_y < h_screen:
                
                if screen.get_at((end_x,end_y)) != (255,255,255):
                    color=screen.get_at((end_x,end_y))
                    self.distance_goal=self.state[0]-end_x
                    self.definition=end_x
                    self.goal_pos=end_x
                    self.visible_=True 
                    
                    pygame.draw.circle(screen, (255, 255, 0), (end_x, end_y), 5)
                    
                        
                # pygame.draw.line(screen,(0,255,0), (sensor_x, sensor_y), (end_x, end_y))
            # pygame.draw.line(screen,(0,255,0), (sensor_x, sensor_y), (end_x, end_y))
            return abs(self.distance_goal)
    def sensor_2(self, num_sensor, w_screen, h_screen, screen):
        """
        Gera sensores à esquerda do agente.

        Args:
            num_sensor (int): Número de sensores.
            w_screen (int): Largura da tela.
            h_screen (int): Altura da tela.
            screen: Superfície da tela.
        
        Returns:
            float: Distância até o objetivo.
        """
        # Cálculos para geração dos sensores à esquerda do agente
        angle_init=math.pi/1
        angle_inc=math.pi*1 /(num_sensor -1)

        com=random.uniform(30,self.vision)

        for i in range(0,num_sensor):
            angle= angle_init + i * angle_inc

            sensor_x=int(self.state[0]+ 20 // 2)
            sensor_y=int(self.state[1]+ 20 // 2)

            end_x = int(sensor_x + com * math.cos(angle)) 
            end_y = int(sensor_y + com * math.sin(angle)) 

            if 0<= end_x < w_screen and 0 <= end_y < h_screen:
                
                if screen.get_at((end_x,end_y)) != (255,255,255):
                    color=screen.get_at((end_x,end_y))
                    self.distance_goal=self.state[0]-end_x
                    self.definition=end_x
                    self.goal_pos=end_x
                    self.visible_=True 
                    
                    pygame.draw.circle(screen, (255, 255, 0), (end_x, end_y), 5)
                  
                # pygame.draw.line(screen,(0,255,0), (sensor_x, sensor_y), (end_x, end_y))
            # pygame.draw.line(screen,(0,255,0), (sensor_x, sensor_y), (end_x, end_y))
            return abs(self.distance_goal)
    # Métodos para tomada de decisão e treinamento do modelo
    def take_d_(self, m_del):
        """        Toma uma decisão sobre o próximo movimento do agente.

        Args:
            m_del: Modelo de classificação.

        Returns:
            str: Ação a ser realizada pelo agente ('left' para esquerda, 'right' para direita).
        """
        # Lógica para a tomada de decisão
        test=[]
        decision=''
        
        dis=abs(self.definition-self.state[0]+self.SPEED)
        
        if (dis)>abs(self.memory_[-1][-1]):#
            test.append([1,1])
            if m_del.predict(test)[-1] == 1:
                decision='right'
        elif (dis)<abs(self.memory_[-1][-1]):
            test.append([0,1])
            if m_del.predict(test)[-1] == 0:
                decision='left'
        else:
    
            mo=['left','right']
            decision=random.choice(mo)
        
        if random.uniform(0,1)<self.epslon or decision=='':
            decision=random.choice(['left','right'])
        
        return decision
    def train_model_(self, inf, data):
        """
        Treina o modelo de classificação.

        Args:
            inf: Informação para treinamento.
            data: Dados para treinamento.
        """
        # Treinamento do modelo
        self.model.fit(data,inf) 
    def predict_(self, new_data):
        """
        Realiza uma previsão com base nos novos dados.

        Args:
            new_data: Novos dados para previsão.

        Returns:
            array: Previsão do modelo.
        """
        return self.model.predict(new_data)

    def acuracy_(self, correct, verify):
        """
        Calcula a acurácia do modelo.

        Args:
            correct: Respostas corretas.
            verify: Respostas verificadas.

        Returns:
            float: Acurácia do modelo.
        """
        return accuracy_score(correct, verify)

    def parse(self):
        """
        Retorna uma ação aleatória.

        Returns:
            str: Ação aleatória ('left' ou 'right').
        """
        dec = ['left', 'right']
        return random.choice(dec)

    def limits(self, w_screen):
        """
        Verifica os limites da tela.

        Args:
            w_screen (int): Largura da tela.
        """
        # Verificação e correção dos limites da tela
        if self.state[0]>=w_screen-30:
            self.state[0]=w_screen-30
        elif self.state[0]<=0:
            self.state[0]=0
    def cal_dist_(self):
        """
        Calcula a distância até o objetivo.

        Returns:
            float: Distância até o objetivo.
        """
        # Cálculo da distância até o objetivo
        if self.definition != 0:
            try:
                return abs(self.definition-self.state[0])
            except Exception as error:
                return 0
                
        else:
            return 0
    def visible_test(self):
        """Função para teste de visibilidade do objetivo (ainda não implementada)."""
        pass

    def update_memory(self, last_act, dis):
        """
        Atualiza a memória do agente.

        Args:
            last_act (str): Última ação realizada pelo agente.
            dis (float): Distância até o objetivo.
        """
        # Atualização da memória do agente
        l_a=0 if last_act=='left' else  1
        try:
            if dis< int(self.memory_[-1][-1]):
                if list([l_a,1]) not in self.table_t:
                    
                    self.table_t.append([l_a,1])
                    self.template.append([l_a])
            if dis> int(self.memory_[-1][-1]):
                if ([l_a,0]) not in self.table_t:
                    
                    self.table_t.append([l_a,0])
                    change=0 if l_a == 1 else 1
                    self.template.append([change])
            
        
            self.train_model_(self.template,self.table_t)
            test=[[1,0]]#caso dê erro aqui, coloque [1,0]
            template_=[0] # e aqui [0]
            p=self.model.predict(test)
            self.rate=accuracy_score(template_,p)
           

           
       
        except Exception as e:
            pass
        self.memory_.append([l_a,dis])
    def text_brain_set(self, text):
        """
        Define o texto exibido na interface gráfica.

        Args:
            text (str): Texto a ser exibido.

        Returns:
            str: Texto ajustado para exibição.
        """
        # Definição do texto exibido na interface gráfica
        if self.visible_:
            return 'Avistei o alvo, irei fazer a classificação das açoes.'
        else:
            if text == '':
                return 'Sem visualização'
            else:
                return text
    def alert(self):
        """Aumenta a visão do agente para identificar o objetivo ."""
        self.vision = 600

    def stagnation(self, count_time, punt_, act, screen, floor):
        """
        Verifica se o agente está em estagnação.

        Args:
            count_time (int): Contagem de tempo.
            punt_ (int): Pontuação atual do agente.
            act (str): Última ação realizada pelo agente.
            screen: Superfície da tela.
            floor (int): Altura do chão na tela.
        """
        # Verificação de estagnação do agente
        tx=5
        self.punt_sec[1]=punt_ if count_time>self.punt_sec[0]+tx else self.punt_sec[1]
        
        print(self.punt_sec, self.vision)
        
        if count_time>self.punt_sec[0]+tx and punt_==self.punt_sec[1] and punt_!=0:
            self.punt_sec[0]=count_time if count_time>self.punt_sec[0]+tx else self.punt_sec[0]
            self.alert()
           
            self.limit+=1
            
        elif self.limit>5:
            self.state[0]=0
            self.limit=0
        else:
            self.vision=350
            
