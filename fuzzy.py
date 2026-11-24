import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np     # criar os universos (ranges), ex.: 0..300, 0..1, etc.
import pygame          # Vetores (pygame.Vector2) - Movimento, direção, normalização
import math            # sqrt para distâncias reais


# ********** THE FUZZY LOGIC ENGINE **********

# Arquitetura da lógica fuzzy: fuzzificação -> regras -> inferência -> defuzzificação.

# Esta classe é o cérebro de um boid -- só calcula força - altera velocidade
class FuzzySystemBoid:
    # Memória temporária global para tratar o caso do líder
    # Guardamos a última lista de boids válida e usamos quando boids vem vazio.
    _GLOBAL_BOIDS_CACHE = []

    # Configuração
    def __init__(self, config):
        self.config = config

        # ==== Setup Variables ====
        # Define inputs/outputs e os seus universos (ranges)
        # Universo (range) --> o intervalo de valores possíveis para cada variável
        self.__setup_variables()

        # ==== Setup Membership Functions ====
        # Define os conjuntos fuzzy (close/mid/far, low/mid/high)
        self.__setup_membership_functions()

        # ==== Setup Rules ====
        # Define as regras IF-THEN (SE-ENTÃO)
        # Nota: o conjunto de regras foi mantido simples: distância ao mais próximo domina a colisão,
        #       e a densidade só reforça separação quando é alta (evita “explosão” de regras).
        self.__setup_rules()

        # ==== Setup Inference System ====
        # Cria o ControlSystem e o ControlSystemSimulation (skfuzzy)
        self.__setup_inference_system()

        # Memória temporária das direções
        # A intensidade vem do fuzzy (sep_strength/eva_strength).
        self._sep_dir = pygame.Vector2(0, 0)
        self._eva_dir = pygame.Vector2(0, 0)

        # Guardar entidade atual (para saber se é leader)
        # É usado no compute() para aplicar boost extra ao leader.
        self._current_entity = None


    # ********** SETUP **********

    def __setup_variables(self):
        # Inputs 
        self.neigh_dist = ctrl.Antecedent(np.arange(0, 301, 1), "neigh_dist")   # distância ao vizinho mais próximo
        self.neigh_count = ctrl.Antecedent(np.arange(0, 21, 1), "neigh_count")  # número de vizinhos no alcance (reforça separação em grupos densos)
        self.pred_dist = ctrl.Antecedent(np.arange(0, 501, 1), "pred_dist")     # distância ao predador (quanto menor -> maior evasão)

        # Outputs         
        self.sep_strength = ctrl.Consequent(np.arange(0, 1.01, 0.01), "sep_strength")   # intensidade da separação 
        self.eva_strength = ctrl.Consequent(np.arange(0, 1.01, 0.01), "eva_strength")   # intensidade da evasão ao predador (depois escalada em força real)

    def __setup_membership_functions(self):
        # - trapmf para estados claros ("muito perto" e "muito longe"): cria patamares
        # - trimf para transições suaves: evita movimentos bruscos
        self.neigh_dist["close"] = fuzz.trapmf(self.neigh_dist.universe, [0, 0, 25, 70])
        self.neigh_dist["mid"]   = fuzz.trimf(self.neigh_dist.universe, [50, 120, 190])
        self.neigh_dist["far"]   = fuzz.trapmf(self.neigh_dist.universe, [160, 230, 300, 300])

        # A densidade reforça separação em zonas muito cheias.
        self.neigh_count["low"]  = fuzz.trapmf(self.neigh_count.universe, [0, 0, 2, 5])
        self.neigh_count["mid"]  = fuzz.trimf(self.neigh_count.universe, [3, 8, 13])
        self.neigh_count["high"] = fuzz.trapmf(self.neigh_count.universe, [10, 14, 20, 20])

        # Quanto mais perto -> evasão mais alta.
        self.pred_dist["close"] = fuzz.trapmf(self.pred_dist.universe, [0, 0, 60, 150])
        self.pred_dist["mid"]   = fuzz.trimf(self.pred_dist.universe, [120, 220, 320])
        self.pred_dist["far"]   = fuzz.trapmf(self.pred_dist.universe, [280, 380, 500, 500])

        # Outputs
        for cons in (self.sep_strength, self.eva_strength):
            cons["low"]  = fuzz.trapmf(cons.universe, [0, 0, 0.2, 0.45])
            cons["mid"]  = fuzz.trimf(cons.universe, [0.25, 0.5, 0.75])
            cons["high"] = fuzz.trapmf(cons.universe, [0.55, 0.8, 1, 1])

    def __setup_rules(self):
        rules = []

        # Separação
        rules += [
            ctrl.Rule(self.neigh_dist["close"], self.sep_strength["high"]),
            ctrl.Rule(self.neigh_dist["mid"], self.sep_strength["mid"]),
            ctrl.Rule(self.neigh_dist["far"], self.sep_strength["low"]),
            ctrl.Rule(self.neigh_count["high"], self.sep_strength["high"]),
        ]

        # Evasão
        rules += [
            ctrl.Rule(self.pred_dist["close"], self.eva_strength["high"]),
            ctrl.Rule(self.pred_dist["mid"], self.eva_strength["mid"]),
            ctrl.Rule(self.pred_dist["far"], self.eva_strength["low"]),
        ]

        self.rules = rules

    def __setup_inference_system(self):
        # Criação do sistema de inferência fuzzy do scikit-fuzzy.
        self.boidz_controller = ctrl.ControlSystem(self.rules)
        self.boidz_sys = ctrl.ControlSystemSimulation(self.boidz_controller)

    # ********** MAIN LOGIC ********** 

    def calculate_fuzzy(self, current_entity, boids: list, predators: list):
        # Guardamos a entidade para saber se é leader no compute()
        self._current_entity = current_entity

        # --- FIX DO LEADER ---
        # Se boids vier vazio, usamos a última lista de boids guardada.
        # Isto evita o leader ficar "cego" e acabar por se sobrepor ao grupo.
        if boids and len(boids) > 0:
            FuzzySystemBoid._GLOBAL_BOIDS_CACHE = boids
        elif FuzzySystemBoid._GLOBAL_BOIDS_CACHE:
            boids = FuzzySystemBoid._GLOBAL_BOIDS_CACHE

        pos = current_entity.position

        # getattr permite aceder a atributos de forma segura.
        # Se o atributo não existir, devolve o valor default (0).
        
        vis2 = getattr(current_entity, "VISUAL_RANGE_SQUARED", 0)
        pred2 = getattr(current_entity, "PREDATOR_EVASION_RANGE_SQUARED", 0)

        # 9999.0 é um valor inicial para indicar "ainda não encontrei o mais próximo".
        # Depois, se continuar 9999.0, substituímos pelo máximo do universo (300).
        nearest_d = 9999.0
        nearest_vec = pygame.Vector2(0, 0)
        count = 0

        # Contar quantos estão dentro do alcance visual (densidade)
        # Encontrar o mais próximo para obter a direção de separação
        for b in boids:
            if b is current_entity:
                continue   # Ignora a si próprio

            dx = pos.x - b.position.x
            dy = pos.y - b.position.y
            d2 = dx*dx + dy*dy

            # Só consideramos boids dentro do alcance visual vis2
            if vis2 and d2 < vis2: # Otimização
                count += 1 # Conta vizinhos para reforçar separação em multidões
                d = math.sqrt(d2)  # Distância real 
                if d < nearest_d:
                    nearest_d = d
                    # nearest_vec aponta para longe do vizinho 
                    nearest_vec = pygame.Vector2(dx, dy)

        # NORMALIZAÇÃO: separa DIREÇÃO (unitária) da INTENSIDADE (fuzzy)
        if nearest_vec.length_squared() > 0:
            self._sep_dir = nearest_vec.normalize()
        else:
            self._sep_dir = pygame.Vector2(0, 0)

        # Se não encontramos vizinhos: usamos 300 (máximo do universo neigh_dist)
        if nearest_d == 9999.0:
            nearest_d = 300.0

        # Predador mais próximo (9999.0 pela mesma lógica do vizinho)
        nearest_pd = 9999.0
        eva_vec = pygame.Vector2(0, 0)

        # EVASÃO: procura predador MAIS PRÓXIMO dentro do alcance de evasão (pred2)
        if predators:
            for p in predators:
                dx = pos.x - p.position.x
                dy = pos.y - p.position.y
                d2 = dx*dx + dy*dy
                if pred2 and 0 < d2 < pred2:
                    d = math.sqrt(d2)  
                    if d < nearest_pd:
                        nearest_pd = d
                        eva_vec = pygame.Vector2(dx, dy)

        if eva_vec.length_squared() > 0:
            self._eva_dir = eva_vec.normalize()
        else:
            self._eva_dir = pygame.Vector2(0, 0)

        # Se não encontramos predador: 500 (máximo do universo pred_dist)
        if nearest_pd == 9999.0:
            nearest_pd = 500.0

        # Inputs crisp do sistema fuzzy:
        # np.clip garante que os valores nunca saem do universo definido.
        self.boidz_sys.input["neigh_dist"] = float(np.clip(nearest_d, 0, 300))
        self.boidz_sys.input["neigh_count"] = float(np.clip(count, 0, 20))
        self.boidz_sys.input["pred_dist"] = float(np.clip(nearest_pd, 0, 500))

        # Executa inferência + defuzzificação e atualiza self.boidz_sys.output
        self.boidz_sys.compute()

    # Decisão do movimento final do boid
    def compute(self):
        # Lê outputs defuzzificados do sistema
        try:
            sep_w = float(self.boidz_sys.output["sep_strength"])   # Intensidade de separação
            eva_w = float(self.boidz_sys.output["eva_strength"])   # Intensidade de evasão
        except Exception:
            # Segurança: se falhar num frame, não aplica força
            return pygame.Vector2(0, 0)

        # BOOST extra para o leader
        # O leader tende a sobrepor-se aos outros boid, este boost ajuda a garantir separação mínima melhor
        leader_boost = 1.4 if getattr(self._current_entity, "is_leader", False) else 1.0

        # Isto cria uma resposta não-linear: muito perto => força cresce mais rapidamente.
        base_sep = ((sep_w * 2.0) + (sep_w**2 * 2.0)) * leader_boost

        # base_eva escala a evasão (multiplicador simples)
        base_eva = eva_w * 1.2

        # Vetor final 
        out = (self._sep_dir * base_sep) + (self._eva_dir * base_eva)

        # Limitamos a magnitude a 3.2 (máximo por frame), isto evita acelerações instantâneas e deixa o movimento mais estável/suave.
        max_step = 3.2
        if out.length() > max_step:
            out.scale_to_length(max_step)

        return out

    # ********** REQUIRED BY ENGINE ********** 

    # Devolve o sistema fuzzy 
    def get_system(self):
        return self.boidz_sys

    def get_output_variables(self) -> list[str]:
        # A interface usa isto para saber que outputs existem e desenhar gráficos.
        if not hasattr(self, 'boidz_controller'):
            return []
        return [consequent.label for consequent in self.boidz_controller.consequents]

    def get_input_variables(self) -> list[str]:
        # A interface usa isto para saber que inputs existem (útil para debug/plots).
        if not hasattr(self, 'boidz_controller'):
            return []
        return [antecedent.label for antecedent in self.boidz_controller.antecedents]

    # ********** PREDATOR FUZZY SYSTEM ********** 

# Controla o comportamento do predador
class FuzzySystemPredator:
    # O predador escolhe uma presa (_target) no calculate_fuzzy() e usa fuzzy para decidir o speed e a turn no compute().
   
    def __init__(self, config):
        self.config = config

        # Pipeline fuzzy: variáveis -> membership -> regras -> inferência
        self.__setup_variables()
        self.__setup_membership_functions()
        self.__setup_rules()
        self.__setup_inference_system()

        # Começa sem alvo para evitar uso de valores antigos
        self._target = None

    # ********** SETUP ********** 

    def __setup_variables(self):
        # Inputs 
        self.prey_dist = ctrl.Antecedent(np.arange(0, 501, 1), "prey_dist")      # distância à presa 
        self.rel_speed = ctrl.Antecedent(np.arange(0, 11, 0.1), "rel_speed")     # velocidade relativa

        # Outputs 
        self.speed_factor = ctrl.Consequent(np.arange(0, 1.01, 0.01), "speed_factor")
        self.turn_factor = ctrl.Consequent(np.arange(0, 1.01, 0.01), "turn_factor")

    def __setup_membership_functions(self):
        # Distância à presa
        self.prey_dist["close"] = fuzz.trapmf(self.prey_dist.universe, [0, 0, 60, 150])
        self.prey_dist["mid"]   = fuzz.trimf(self.prey_dist.universe, [120, 220, 320])
        self.prey_dist["far"]   = fuzz.trapmf(self.prey_dist.universe, [280, 380, 500, 500])

        # Velocidade relativa
        self.rel_speed["low"]   = fuzz.trapmf(self.rel_speed.universe, [0, 0, 1, 2.5])
        self.rel_speed["mid"]   = fuzz.trimf(self.rel_speed.universe, [2, 4, 6])
        self.rel_speed["high"]  = fuzz.trapmf(self.rel_speed.universe, [5.5, 7, 10, 10])

        # Mantemos a mesma forma usada no boid para consistência e transições suaves.
        for cons in (self.speed_factor, self.turn_factor):
            cons["low"]  = fuzz.trapmf(cons.universe, [0, 0, 0.2, 0.45])
            cons["mid"]  = fuzz.trimf(cons.universe, [0.25, 0.5, 0.75])
            cons["high"] = fuzz.trapmf(cons.universe, [0.55, 0.8, 1, 1])

    def __setup_rules(self):
       # Regras:
        self.rules = [
            ctrl.Rule(self.prey_dist["close"], self.speed_factor["high"]),
            ctrl.Rule(self.prey_dist["mid"], self.speed_factor["mid"]),
            ctrl.Rule(self.prey_dist["far"], self.speed_factor["low"]),
            ctrl.Rule(self.rel_speed["high"], self.turn_factor["high"]),
            ctrl.Rule(self.rel_speed["mid"], self.turn_factor["mid"]),
            ctrl.Rule(self.rel_speed["low"], self.turn_factor["low"]),
        ]

    def __setup_inference_system(self):
        # Cria o sistema de inferência fuzzy. 
        self.boidz_controller = ctrl.ControlSystem(self.rules)
        self.boidz_sys = ctrl.ControlSystemSimulation(self.boidz_controller)

    # ********** MAIN LOGIC ********** 

    # Qual boid vai ser perseguido pelo predador 
    def calculate_fuzzy(self, current_entity, boids: list):
        # Reset ao target a cada frame (a presa pode mudar)
        self._target = None

        # Definição do alcance de caça
        hunt2 = getattr(current_entity, "hunting_range_squared", None)
        if hunt2 is None:
            vr = getattr(current_entity, "VISUAL_RANGE", 0)
            hunt2 = vr * vr

        min_d2 = float("inf")
        # Procurar boid mais próximo dentro do alcance de caça.
        for b in boids:
            # Distância entre predador e boid 
            dx = current_entity.position.x - b.position.x 
            dy = current_entity.position.y - b.position.y
            d2 = dx*dx + dy*dy   # Distância ao quadrado
            # Se é o mais proximo e se está no alcance de caça
            if d2 < min_d2 and d2 < hunt2:
                min_d2 = d2
                self._target = b

        # Se não há alvo:
        # definimos inputs neutros (presa distante e rel_speed=0) para outputs baixos, e saímos cedo.
        if self._target is None:
            # Evita comportamentos desnecessários 
            self.boidz_sys.input["prey_dist"] = 500  
            self.boidz_sys.input["rel_speed"] = 0
            self.boidz_sys.compute()
            return

        # dist (sqrt) é necessário porque prey_dist é crisp em unidades reais (pixels)
        dist = math.sqrt(min_d2)

        # rel_speed mede quão diferente é a velocidade (norma do vetor diferença)
        rel = (self._target.velocity - current_entity.velocity).length()  # Calculo da diferença entre a velocidade do predador e da presa

        # Clipping para respeitar universos
        self.boidz_sys.input["prey_dist"] = float(np.clip(dist, 0, 500))
        self.boidz_sys.input["rel_speed"] = float(np.clip(rel, 0, 10))
        self.boidz_sys.compute()

    def compute(self, current_entity):
        # Sem alvo não há steering (não há caça)
        if self._target is None:
            return pygame.Vector2(0, 0)

        # Outputs defuzzificados (0..1)
        speed_f = float(self.boidz_sys.output["speed_factor"])
        turn_f = float(self.boidz_sys.output["turn_factor"])

        # desired aponta do predador para a presa
        desired = (self._target.position - current_entity.position)
        if desired.length_squared() > 0:
            desired = desired.normalize()

        # predator_speed vem do Predator (config)
        predator_speed = getattr(current_entity, "predator_speed", 2.0)

        # - 0.7*speed_f aumenta agressividade conforme o fuzzy
        desired_velocity = desired * predator_speed * (0.7 + 0.7 * speed_f)

        # Isto é o método clássico de steering behaviours (ajusta a trajetória gradualmente).
        steering = desired_velocity - current_entity.velocity

        # max_force vem do Predator; escalamos com turn_factor para permitir virar mais/menos agressivo
        max_force = getattr(current_entity, "max_force", 0.5) * (0.6 + 0.8 * turn_f)

        # Limitamos a força máxima aplicada (evita mudanças instantâneas/instabilidade)
        if steering.length_squared() > max_force * max_force:
            steering.scale_to_length(max_force)

        return steering

    # ********** REQUIRED BY ENGINE ********** 

    def get_system(self):
        return self.boidz_sys

    def get_output_variables(self) -> list[str]:
        # Para plots/UI
        if not hasattr(self, 'boidz_controller'):
            return []
        return [consequent.label for consequent in self.boidz_controller.consequents]

    def get_input_variables(self) -> list[str]:
        # Para plots/UI
        if not hasattr(self, 'boidz_controller'):
            return []
        return [antecedent.label for antecedent in self.boidz_controller.antecedents]
    