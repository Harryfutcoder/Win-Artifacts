import random

import pymysql

import multi_agent.multi_agent_system
from action.web_action import WebAction
from state.impl.action_set_with_execution_times_state import ActionSetWithExecutionTimesState
from state.web_state import WebState


class QTableDB:
    def __init__(self, host, user, password, database):
        self.conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4'
        )
        self.cursor = self.conn.cursor()
        self.create_table_if_not_exists()

    def create_table_if_not_exists(self):
        drop_table_sql = "DROP TABLE IF EXISTS q_values;"
        create_table_sql = """
        CREATE TABLE q_values (
            agent_name VARCHAR(255),
            state INTEGER,
            action INTEGER,
            q_value DOUBLE,
            PRIMARY KEY (agent_name, state, action)
        );
        """
        self.cursor.execute(drop_table_sql)
        self.cursor.execute(create_table_sql)
        self.conn.commit()

    def get_q_value(self, agent_name, state, action):
        sql = "SELECT q_value FROM q_values WHERE agent_name=%s AND state=%s AND action=%s"
        self.cursor.execute(sql, (agent_name, state, action))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def set_q_value(self, agent_name, state, action, q_value):
        if self.get_q_value(agent_name, state, action) is None:
            sql = "INSERT INTO q_values (agent_name, state, action, q_value) VALUES (%s, %s, %s, %s)"
            self.cursor.execute(sql, (agent_name, state, action, q_value))
        else:
            sql = "UPDATE q_values SET q_value=%s WHERE agent_name=%s AND state=%s AND action=%s"
            self.cursor.execute(sql, (q_value, agent_name, state, action))
        self.conn.commit()

    def get_all_q_values(self, agent_name, state):
        sql = "SELECT action, q_value FROM q_values WHERE agent_name=%s AND state=%s"
        self.cursor.execute(sql, (agent_name, state))
        return dict(self.cursor.fetchall())

    def get_best_actions(self, agent_name, state):
        q_values = self.get_all_q_values(agent_name, state)
        if not q_values:
            return [], -9999
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return best_actions, max_q


class MargDB(multi_agent.multi_agent_system.MultiAgentSystem):
    def __init__(self, params):
        super(MargDB, self).__init__(params)
        self.params = params
        self.agent_type = params["agent_type"]
        self.epsilon = params["epsilon"]
        self.initial_q_value = params["initial_q_value"]
        self.gamma = params["gamma"]
        self.alpha = params["alpha"]
        self.R_PENALTY = -9999

        self.qdb = QTableDB(
            host='localhost',
            user='root',
            password='123456',
            database='webtest'
        )
        self.state_list = []
        self.action_list = []

    def get_action_algorithm(self, web_state: WebState, html: str, agent_name: str) -> WebAction:
        self.update(web_state, html, agent_name)
        actions = web_state.get_action_list()

        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)

        if self.agent_type == 'cql':
            state_index = self.state_list.index(web_state)
            best_actions, _ = self.qdb.get_best_actions('cql', state_index)
        else:
            q_values = {}
            for i in range(self.agent_num):
                agent_id = str(i)
                state_index = self.state_list.index(web_state)
                values = self.qdb.get_all_q_values(agent_id, state_index)
                for action, q in values.items():
                    q_values[action] = q_values.get(action, 0.0) + q
            if not q_values:
                return random.choice(actions)
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
        action_index = random.choice(best_actions) if best_actions else random.choice(actions)
        return self.action_list[action_index]

    def get_reward(self, action: WebAction, state: WebState) -> float:
        actions = state.get_action_list()
        if len(actions) == 0:
            return self.R_PENALTY
        if isinstance(state, ActionSetWithExecutionTimesState):
            assert self.action_dict[action] is not None
            return 1.0 if self.action_dict[action] == 0 else 1.0 / float(self.action_dict[action])
        return self.R_PENALTY

    def update(self, web_state: WebState, html: str, agent_name: str):
        if not self.state_list.__contains__(web_state):
            self.state_list.append(web_state)
        actions = web_state.get_action_list()
        for temp_action in actions:
            if not self.action_list.__contains__(temp_action):
                self.action_list.append(temp_action)
        for action in actions:
            state_index = self.state_list.index(web_state)
            action_index = self.action_list.index(action)
            if self.qdb.get_q_value(agent_name, state_index, action_index) is None:
                self.qdb.set_q_value(agent_name, state_index, action_index, self.initial_q_value)

        if self.prev_action_dict[agent_name] is None or self.prev_state_dict[agent_name] is None:
            return
        if not isinstance(self.prev_state_dict[agent_name], ActionSetWithExecutionTimesState):
            return

        reward = self.get_reward(self.prev_action_dict[agent_name], web_state)
        prev_state = self.prev_state_dict[agent_name]
        prev_action = self.prev_action_dict[agent_name]

        if self.agent_type == 'cql':
            state_index = self.state_list.index(web_state)
            _, max_q_value = self.qdb.get_best_actions('cql', state_index)
            prev_state_index = self.state_list.index(prev_state)
            prev_action_index = self.action_list.index(prev_action)
            q_predict = self.qdb.get_q_value('cql', prev_state_index, prev_action_index) or self.initial_q_value
            q_target = reward + self.gamma * max_q_value
            new_q = q_predict + self.alpha * (q_target - q_predict)
            self.qdb.set_q_value('cql', prev_state_index, prev_action_index, new_q)
            print(
                f"[CQL] Thread {agent_name}: Q({prev_state_index}, {prev_action_index}) updated {q_predict} -> {new_q}")

        else:
            total_q = 0
            contributing_agents = 0
            for i in range(self.agent_num):
                other_agent = str(i)
                if other_agent == agent_name:
                    continue
                state_index = self.state_list.index(web_state)
                values = self.qdb.get_all_q_values(other_agent, state_index)
                if values:
                    contributing_agents += 1
                    _, max_value = self.qdb.get_best_actions(other_agent, state_index)
                    total_q += max_value

            prev_state_index = self.state_list.index(prev_state)
            prev_action_index = self.action_list.index(prev_action)
            q_predict = self.qdb.get_q_value(agent_name, prev_state_index, prev_action_index) or self.initial_q_value
            if contributing_agents > 0:
                avg_q = total_q / contributing_agents
                new_q = q_predict + self.alpha * reward + self.gamma * (avg_q - q_predict)
            else:
                state_index = self.state_list.index(web_state)
                _, max_q_value = self.qdb.get_best_actions(agent_name, state_index)
                q_target = reward + self.gamma * max_q_value
                new_q = q_predict + self.alpha * (q_target - q_predict)

            self.qdb.set_q_value(agent_name, prev_state_index, prev_action_index, new_q)
            print(
                f"[DQL] Thread {agent_name}: Q({prev_state_index}, {prev_action_index}) updated {q_predict} -> {new_q}")
