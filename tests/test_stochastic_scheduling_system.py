import ssenvs.envs.instances as jsslib
from ssenvs.envs.poss import StochasticEnv
import logging
import numpy as np
import rustworkx as rx

logger = logging.getLogger(__name__)
logger.propagate = True

def test_sim_ta01(caplog) -> None:
    #logging.basicConfig(level=logging.DEBUG, filename='./logs/test_sim_ta01.log')
    caplog.set_level(logging.DEBUG)
    logger.info("Started simulation")

    env = StochasticEnv(spec=dict(instance_path=jsslib.get_path() / "ta01"))

    obs0, info0 = env.reset(seed=42)

    logger.info(f"Pending jobs at t=0: {obs0.get('state').pending}")
    assert len(obs0.get('state').pending)==0
    print(env.releases)
    print(env.t_max)

    t: int = 0

    for _ in range(10):
        obs_t, info_t, r_t, done_t, trunc_t = env.step([])

        s_t = obs_t.get('state')
        A_t = info_t.get('arms')
        r_t = info_t.get('feedback')

        print(f"t: {t}, |P_t| = {len(s_t.pending)}, |A_t| = {len(A_t)}, |r_t| = {len(r_t)}")
        print(f"P_t: {s_t.pending}")
        print(f"W_t: {s_t.working}")
        print(f"I_t: {s_t.idle}")
        if t >= 7:
            assert len(s_t.pending) > 0
            assert len(A_t) > 0

        assert len(r_t) == 0
        t += 1

    # Apply one action

    while True:
        GA_t = rx.PyGraph()
        J = set()
        I = set()
        for j, i in A_t:
            J.add(j)
            I.add(i)
        vertex_ids = dict()
        rev_vertex_ids = dict()
        edge_ids = dict()
        for j in J:
            id = GA_t.add_node((0, j))
            vertex_ids[(0, j)] = id
            rev_vertex_ids[id] = (0, j)
        for i in I:
            id = GA_t.add_node((1, i))
            vertex_ids[(1, i)] = id
            rev_vertex_ids[id] = (1, i)
        for j, i in A_t:
            GA_t.add_edge(vertex_ids[(0, j)], vertex_ids[(1, i)], None)
        m_t = rx.max_weight_matching(GA_t)
        print(m_t)
        a_t = []
        for u, v in m_t:
            u1 = rev_vertex_ids[u]
            v1 = rev_vertex_ids[v]
            if u1[0] == 0:
                a_t.append((u1[1], v1[1]))
            else:
                a_t.append((v1[1], u1[1]))

        print(f"a_t: {a_t}")
        obs_t, info_t, r_t, done_t, trunc_t = env.step(a_t)

        s_t = obs_t.get('state')
        A_t = info_t.get('arms')
        r_t = info_t.get('feedback')

        print(f"t: {t}, |P_t| = {len(s_t.pending)}, |A_t| = {len(A_t)}, |r_t| = {len(r_t)}")
        print(f"P_t: {s_t.pending}")
        print(f"W_t: {s_t.working}")
        print(f"I_t: {s_t.idle}")

        if len(r_t) > 0:
            print(r_t)
            break
        t+=1

    assert t == 71
    assert r_t[(3, 12)] == 1.0

    logger.info("Finished simulation")