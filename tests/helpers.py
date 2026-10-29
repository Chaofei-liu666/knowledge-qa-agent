"""测试用的确定性策略：按脚本回放决策。"""


class Scripted:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.i = 0

    def decide(self, state):
        d = self.decisions[min(self.i, len(self.decisions) - 1)]
        self.i += 1
        return d
