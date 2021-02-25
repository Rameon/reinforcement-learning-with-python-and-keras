 # -*- coding: utf-8 -*-
from environment import GraphicDisplay, Env


class ValueIteration:
    def __init__(self, env):
        # 환경 객체 생성
        self.env = env
        # 가치 함수를 2차원 리스트로 초기화
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # 감가율
        self.discount_factor = 0.9

    # 가치 이터레이션
    # 정책 이터레이션과 가치 이터레이션의 중요한 차이점 : 정책 이터레이션에서 정책 평가와 정책 발전으로 단계가 나누어져 있다면,
    #                                             가치 이터레이션은 현재의 가치 함수가 최적 정책에 대한 가치함수라고 가정하기 때문에
    #                                             정책을 발전하는 함수가 필요없으므로 ValueIteration 클래스는 좀 더 간단해짐
    #                                             -> 현재 가치함수를 바탕으로 최적 행동을 반환하는 get_action() 함수를 정책을 출력하는데 대신 사용함
    #                                             정책 이터레이션의 정책 평가와 정책 발전이 value_iteration() 함수 하나로 대체됨
    #                                             정책이 독립적으로 존재하지 않기 때문에 get_policy 함수가 없음
    # 벨만 최적 방정식을 통해 다음 가치 함수 계산
    def value_iteration(self):
        next_value_table = [[0.0] * self.env.width for _ in
                            range(self.env.height)]
        for state in self.env.get_all_states():
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            # 가치 함수를 위한 빈 리스트
            value_list = []

            # 가능한 모든 행동에 대해 계산
            # 벨만 최적 방정식에서는 max를 계산해야하므로 모든 행동에 대해 계산하고,
            # (reward + self.discount_factor * next_value) 값을 value_list에 저장함
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))
            # 최댓값을 다음 가치 함수로 대입
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
        self.value_table = next_value_table

    # 현재 가치 함수로부터 행동을 반환
    def get_action(self, state):
        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        # 모든 행동에 대해 큐함수 (보상 + (할인율 * 다음 상태 가치함수))를 계산
        # 최대 큐 함수를 가진 행동(복수일 경우 여러 개)을 반환
        for action in self.env.possible_actions:

            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            # 벨만 최적 방정식을 통해 구한 가치함수를 토대로, 에이전트는 자신이 할 행동을 구할 수 있음
            # 최적 정책이 아니더라도, 사용자는 현재 가치함수에 대한 탐욕 정책을 볼 수 있음
            # 탐욕 정책을 위해서는 큐함수를 비교해야 하므로, 모든 행동에 대해 아래 코드를 실행해 큐함수를 구함
            value = (reward + self.discount_factor * next_value)

            # 그중에서 가장 큰 value 값을 가지는 행동의 인덱스를 가져옴
            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        return action_list

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()