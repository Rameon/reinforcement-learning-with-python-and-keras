# -*- coding: utf-8 -*-
import random

# environment.py 안에 있는 GraphicDisplay, Env 클래스를 import함
# GraphicDisplay 는 GUI로 그리드월드 환경을 보여주는 클래스임
from environment import GraphicDisplay, Env


# 정책 이터레이션의 에이전트는 PolicyIteration 클래스로 정의되어 있음
class PolicyIteration:
    def __init__(self, env):  # PolicyIteration 클래스의 정의에서 env를 self.env로서 정의함
        # 환경에 대한 객체 선언
        # 에이전트에게는 환경에 대한 정보가 필요하므로 main 루프에서 Env()를 env 객체로 생성함
        # 이 env 객체를 PolicyIteration 클래스의 인수로 전달함으로써, 에이전트는 환경의 Env() 클래스에 접근할 수 있음
        self.env = env

        # < env 객체에 정의돼 있는 변수와 함수 >
        # env.width, env.height : 그리드 월드의 너비, 높이 -> 반환값 : 그리드월드의 가로, 세로를 정수로 반환함

        # env.state_after_action(state, action) : 에이전트가 특정 상태에서 특정 행동을 했을 때, 에이전트가 가는 다음 상태
        #                                         -> 반환값 : 행동 후의 상태를 좌표로 표현한 리스트를 반환함 예) [1,2]

        # DP에서는 에이전트가 모든 상태에 대해 벨만 방정식을 계산하는데, 따라서 에이전트는 가능한 모든 상태를 알아야 함
        # 이 모든 상태들은 env.get_all_state() 를 통해 할 수 있음

        # env.get_all_state() : 존재하는 모든 상태 -> 반환값 : 모든 상태를 반환함 예) [[0,0], [0,1], ... , [4,4]]

        # env.get_reward(state, action) : 특정 상태의 보상(환경이 주는 보상) -> 반환값 : 정수의 형태로 보상을 반환함

        # env.possible_actions : 상, 하, 좌, 우(에이전트의 가능한 모든 행동) -> 반환값 : [0,1,2,3]을 반환함, 순서대로 상,하,좌,우를 의미함

        # 보상과 상태 변환 확률은 에이전트가 아니라, 환경에 속한 것이므로 env 객체로 정의함

        # 가치함수를 2차원 리스트로 초기화
        # 정책 이터레이션은 모든 상태에 대해 가치함수를 계산하기 때문에 value_table 이라는 2차원 리스트 변수로 가치함수를 선언함
        # 그리드 월드의 환경은 세로 5칸, 가로 5칸의 크기를 가지므로 5X5의 2차원 리스트가 됨
        # 모든 상태의 가치함수의 값을 0으로 초기화함
        self.value_table = [[0.0] * env.width for _ in range(env.height)]

        # 상 하 좌 우 동일한 확률로 정책 초기화
        # 정책 policy_table 은 모든 상태에 대해 상, 하, 좌, 우에 해당하는 각 행동의 확률을 담고 있는 리스트임
        # 따라서 5X5X4의 3차원 리스트로 생성함
        # 정책은 무작위 정책으로 초기화 하는데, 이때 상,하,좌,우 행동의 확률이 25%임
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width
                                    for _ in range(env.height)]
        # 마침 상태의 설정
        self.policy_table[2][2] = []
        # 감가율(할인율)
        self.discount_factor = 0.9

    # 정책 평가
    # DP에서는 사용자가 주는 입력에 따라 에이전트가 역할을 수행하기 때문에,
    # 에이전트는 environment.py의 GraphicDisplay 클래스에서 실행됨
    # 따라서 GraphicDisplay 클래스는 PolicyIteration 클래스의 객체인 policy_iteration을 상속받음
    def policy_evaluation(self):
        # 다음 가치함수 초기화
        # 정책 평가에서 에이전트는 모든 상태의 가치함수를 업데이트함
        # 이를 위해 next_value_table 을 선언하고, 계산 결과를 여기에 저장함
        # 모든 상태에 대해 벨만 기대 방정식의 계산이 끝나면 현재의 value_table에 next_value_table을 덮어쓰는 식으로 정책 평가를 진행함
        next_value_table = [[0.00] * self.env.width
                                    for _ in range(self.env.height)]

        # 모든 상태에 대해서 벨만 기대방정식을 계산
        for state in self.env.get_all_states():
            value = 0.0
            # 마침 상태의 가치 함수 = 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue

            # 벨만 기대 방정식
            for action in self.env.possible_actions:
                # 행동을 취했을 경우, 다음 상태가 어딘지 알려주는 역할을 하는 것 : env.state_after_action(state, action)
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                # 벨만 기대 방정식을 계산하는 부분
                # get_policy 함수를 통해 각 상태에서 각 행동에 대한 확률값을 구함
                # 다음 상태로 갔을 때 받을 보상과 다음 상태의 가치함수를 할인해서 더함
                # 정책이 각 행동에 대한 확률을 나타내기 때문에 모든 행동에 대해 value를 계산하고 더하면 기댓값을 계산한 것이 됨
                value += (self.get_policy(state)[action] *
                          (reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

    # 정책 개선
    # 현재 가치 함수에 대해서 탐욕 정책 발전
    # 정책 평가를 통해 정책을 평가하면 그에 따른 새로운 가치함수를 얻음
    # 에이전트는 새로운 가치함수를 통해 정책을 업데이트함
    def policy_improvement(self):
        next_policy = self.policy_table  # 정책 발전에서 정책 policy_table을 복사한 next_policy에 업데이트된 정책을 저장함
        # 정책을 업데이트하는 방법 중에서 탐욕 정책 발전을 사용함
        # 탐욕 정책 발전 : 가치가 가장 높은 하나의 행동을 선택하는 것
        #               현재 상태에서 가장 좋은 행동이 여러 개일 수도 있는데, 가장 좋은 행동들을 동일한 확률로 선택하는 정책으로 업데이트함
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue

            value = -99999

            max_index = []
            # 반환할 정책 초기화
            result = [0.0, 0.0, 0.0, 0.0]

            # 모든 행동에 대해서 [보상 + (감가율 * 다음 상태 가치함수)] 계산함
            # 계산한 값은 max_index 리스트에 저장함
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                temp = reward + self.discount_factor * next_value

                # 받을 보상이 최대인 행동의 index(최대가 복수라면 모두)를 추출
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            # 행동의 확률 계산
            # max_index 에 담긴 값이 여러 개라면 에이전트는 max_index에 담긴 index의 행동들을 동일한 확률에 기반해서 선택함
            # 이를 구현하기 위해 1을 max_index의 길이로 나눠서 행동의 확률을 계산함
            prob = 1 / len(max_index)

            # max_index의 index에 해당하는 행동에 계산한 확률값을 저장함
            for index in max_index:
                result[index] = prob

            next_policy[state[0]][state[1]] = result

        self.policy_table = next_policy

    # 특정 상태에서 정책에 따른 행동을 반환
    # 에이전트가 정책에 따라서 움직이려면 특정 상태에서 어떤 행동을 해야 할지 알아야 하고, 이 역할을 하는 것이 get_action 함수임
    def get_action(self, state):
        # 0 ~ 1 사이의 값을 무작위로 추출
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0
        # 정책에 담긴 행동 중에 무작위로 한 행동을 추출
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    # 상태에 따른 정책 반환
    # self.policy_table로 저장돼 있는 정책에서 해당 상태에 대한 정책을 반환함
    def get_policy(self, state):
        if state == [2, 2]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    # 가치 함수의 값을 반환
    # self.value_table로 저장돼 있는 가치함수에서 해당 상태에 해당하는 가치함수를 반환함
    def get_value(self, state):
        # 소숫점 둘째 자리까지만 계산
        return round(self.value_table[state[0]][state[1]], 2)


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()