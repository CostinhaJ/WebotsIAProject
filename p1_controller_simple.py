import numpy as np
from controller import Supervisor
import random
import math
import matplotlib.pyplot as plt

# ================= CONFIGURAÇÕES OTIMIZADAS =================
TIME_STEP      = 64    # passo de física Webots (ms)
POPULATION_SIZE= 20
GENERATIONS    = 30
PARENTS_KEEP   = 4
MUTATION_RATE  = 0.2
MUTATION_SIZE  = 0.15
EVALUATION_TIME= 20    # segundos simulados
MAX_SPEED      = 8.0

INPUT_SIZE     = 5   # 2 chão + 3 frontais
HIDDEN_SIZE    = 5
OUTPUT_SIZE    = 2
GENOME_SIZE    = (INPUT_SIZE+1)*HIDDEN_SIZE + (HIDDEN_SIZE+1)*OUTPUT_SIZE

# ================= CONTROLLER =================
class ThymioController:
    def __init__(self):
        self.sup            = Supervisor()
        self.robot_node     = self.sup.getFromDef("ROBOT")
        self.translation_f  = self.robot_node.getField("translation")
        self.rotation_f     = self.robot_node.getField("rotation")
        self.timestep       = int(self.sup.getBasicTimeStep())

        # motores
        self.left_motor  = self.sup.getDevice('motor.left')
        self.right_motor = self.sup.getDevice('motor.right')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))

        # sensores
        self.ground = [
            self.sup.getDevice('prox.ground.0'),
            self.sup.getDevice('prox.ground.1')
        ]
        self.front  = [
            self.sup.getDevice('prox.horizontal.0'),
            self.sup.getDevice('prox.horizontal.2'),
            self.sup.getDevice('prox.horizontal.4'),
        ]
        for s in self.ground + self.front:
            s.enable(self.timestep)

    def reset(self, fix_position=False):
        # rotação apenas em torno do eixo vertical
        yaw = 0.0 if fix_position else random.uniform(0, 2*math.pi)
        self.rotation_f.setSFRotation([0,0,1,yaw])
        # posição ligeiramente elevada
        self.translation_f.setSFVec3f([0,0.03,0])
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.sup.step(self.timestep)
        # armazena posição inicial
        self.prev_pos = self.robot_node.getPosition()

    def read_sensors(self):
        # normaliza chão: preto ≈1, branco ≈0
        ground_vals = [1 - (s.getValue()/1023.0) for s in self.ground]
        # normaliza frontais: perto ≈1, longe ≈0 (max ~4096)
        front_vals  = [min(s.getValue()/2000.0,1.0) for s in self.front]
        return ground_vals + front_vals

    def step(self, genome):
        # lê sensores
        inputs = self.read_sensors()
        on_line = any(v > 0.5 for v in inputs[:2])
        # detecta colisão se frontal ≳0.8
        if any(v > 0.8 for v in inputs[2:]):
            return False

        # ANN feed‑forward
        iv = np.array(inputs + [1.0])
        w1 = np.array(genome[: (INPUT_SIZE+1)*HIDDEN_SIZE]).reshape((HIDDEN_SIZE, INPUT_SIZE+1))
        w2 = np.array(genome[(INPUT_SIZE+1)*HIDDEN_SIZE:]).reshape((OUTPUT_SIZE, HIDDEN_SIZE+1))
        h  = np.tanh(w1.dot(iv))
        ov = np.tanh(w2.dot(np.concatenate([h,[1.0]])))

        # calcula velocidades
        left_v  = max(min(ov[0]*MAX_SPEED, MAX_SPEED), -MAX_SPEED)
        right_v = max(min(ov[1]*MAX_SPEED, MAX_SPEED), -MAX_SPEED)
        self.left_motor.setVelocity(left_v)
        self.right_motor.setVelocity(right_v)

        # avança simulação
        self.sup.step(self.timestep)

        # calcula distância para fitness
        curr = self.robot_node.getPosition()
        dx, dz = curr[0]-self.prev_pos[0], curr[2]-self.prev_pos[2]
        dist = math.hypot(dx, dz)
        self.prev_pos = curr

        return dist, on_line

    def evaluate(self, genome, fix_position=False):
        self.reset(fix_position)
        start = self.sup.getTime()
        fitness = 0.0
        line_time = 0.0

        while self.sup.getTime() - start < EVALUATION_TIME:
            result = self.step(genome)
            if result is False:
                # colisão
                fitness -= 20
                break
            dist, on_line = result
            if on_line:
                fitness += dist * 3 + 0.05
                line_time += self.timestep/1000.0
            else:
                fitness += dist * 0.5
        # bônus final por tempo na linha
        fitness += line_time * 5.0
        return fitness

# ================= EVOLUÇÃO =================
def run_evolution():
    ctrl = ThymioController()
    # população inicial (valores pequenos em [-1,1])
    population = [np.random.uniform(-1,1,GENOME_SIZE).tolist()
                  for _ in range(POPULATION_SIZE)]
    best_genome = None
    best_fitness= -1e9
    history = []

    print(f"=== Evolução: {GENERATIONS} gerações, pop={POPULATION_SIZE} ===")
    for gen in range(1, GENERATIONS+1):
        fitnesses = []
        for i, ind in enumerate(population):
            f = ctrl.evaluate(ind)
            fitnesses.append(f)
        # ordena
        idx         = np.argsort(fitnesses)[::-1]
        population  = [population[i] for i in idx]
        fitnesses   = [fitnesses[i] for i in idx]
        gen_best    = fitnesses[0]
        history.append(gen_best)

        if gen_best > best_fitness:
            best_fitness = gen_best
            best_genome  = population[0]

        print(f"Ger {gen:02d}: best_gen={gen_best:.1f}, best_all={best_fitness:.1f}")

        # elitismo
        next_pop = population[:PARENTS_KEEP]
        # crossover + mutação
        while len(next_pop) < POPULATION_SIZE:
            p1, p2 = random.sample(population[:PARENTS_KEEP],2)
            # one‑point crossover
            pt = random.randint(1,GENOME_SIZE-1)
            child = p1[:pt] + p2[pt:]
            # mutação
            child = [g + np.random.normal(0,MUTATION_SIZE) if random.random()<MUTATION_RATE else g
                     for g in child]
            next_pop.append(child)
        population = next_pop

    # reavaliação robusta do melhor
    print("\n=== Teste final do melhor indivíduo ===")
    for i in range(5):
        f = ctrl.evaluate(best_genome, fix_position=True)
        print(f" Teste {i+1}: {f:.1f}")

    # plot histórico
    plt.plot(history, '-o')
    plt.title("Melhor fitness por geração")
    plt.xlabel("Geração")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.savefig("fitness_evolution.png")
    print("Gráfico salvo → fitness_evolution.png")

    return best_genome

if __name__ == "__main__":
    run_evolution()
