// handwired × sprite 最小示範：神經節律 → 肌肉 → 關節 → 骨架 → 像素
// 執行：node examples/sprite-arm-chain.mjs
// 每一步只用乘、加、ReLU（sin/cos 由狀態 (c, s) 以小角度矩陣更新，全程線性）。

const relu = x => Math.max(0, x);

// ---------- L0：Matsuoka CPG（兩個互相抑制的神經元，輸出本身就是 ReLU） ----------
const cpg = {x1: 0.1, x2: 0, v1: 0, v2: 0};
const CPG = {tau: 0.25, T: 0.5, beta: 2.5, w: 2.0, s: 1.0};
function stepCPG(dt) {
    const y1 = relu(cpg.x1), y2 = relu(cpg.x2);
    cpg.x1 += (dt / CPG.tau) * (-cpg.x1 - CPG.beta * cpg.v1 - CPG.w * y2 + CPG.s);
    cpg.x2 += (dt / CPG.tau) * (-cpg.x2 - CPG.beta * cpg.v2 - CPG.w * y1 + CPG.s);
    cpg.v1 += (dt / CPG.T) * (-cpg.v1 + y1);
    cpg.v2 += (dt / CPG.T) * (-cpg.v2 + y2);
    return y1 - y2; // 屈肌 − 伸肌
}

// ---------- L1：肌肉（激活動力學 + 增益，線性） ----------
const joints = [ // 骨長取 Winter 比例（身高 H = 30 像素）；lo/hi 為關節限位（弧度）
    {name: '上臂', len: 0.186 * 30, gain: 40, lo: -1.2, hi: 1.2, delay: 0},
    {name: '前臂', len: 0.146 * 30, gain: 30, lo: 0, hi: 2.4, delay: 6},
    {name: '手', len: 0.108 * 30, gain: 15, lo: -0.8, hi: 0.8, delay: 12}
].map(j => ({...j, a: 0, th: 0, om: 0}));
const history = []; // 傳導延遲：越遠的關節，收到的訊號越晚

// ---------- L2：關節動力學（彈簧阻尼 + ReLU 軟限位，半隱式 Euler） ----------
const K = 20, C = 2 * Math.sqrt(K), KLIM = 200, TAU_ACT = 0.05;
function stepJoint(j, u, dt) {
    j.a += (u - j.a) * dt / TAU_ACT;
    const tau = j.gain * j.a - K * j.th - C * j.om
        - KLIM * relu(j.th - j.hi) + KLIM * relu(j.lo - j.th);
    j.om += tau * dt;
    j.th += j.om * dt;
}

// ---------- L3：前向運動學（矩陣連乘：越末端，位移是越多段旋轉的疊加） ----------
function fk(shoulder = [10, 8]) {
    let [x, y] = shoulder, c = 0, s = 1; // 初始朝下：(cos, sin) = (0, 1)
    const pts = [[x, y]];
    for (const j of joints) {
        const cj = Math.cos(j.th), sj = Math.sin(j.th); // θ 視為外部參數，R 在層外算好
        [c, s] = [c * cj - s * sj, s * cj + c * sj];   // R_total = R_parent · R(θ)
        x += j.len * c; y += j.len * s;
        pts.push([x, y]);
    }
    return pts;
}

// ---------- L5：像素化（沿骨頭取樣，ReLU 階梯取整） ----------
// 每個 k 一個陡峭階梯：在 x = k + 0.5 處從 0 跳到 1（寬度 EPS），全部加總即為四捨五入
const EPS = 1e-6;
const roundStep = x => { let r = -40; for (let k = -40; k < 40; k++) r += (relu(x - k - 0.5 + EPS) - relu(x - k - 0.5)) / EPS; return Math.round(r * 1e6) / 1e6; };
function render(pts, W = 24, H = 22) {
    const g = Array.from({length: H}, () => Array(W).fill('·'));
    const marks = ['█', '▓', '▒'];
    for (let i = 0; i < pts.length - 1; i++) {
        for (let t = 0; t <= 1; t += 0.1) {
            const px = roundStep(pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t);
            const py = roundStep(pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t);
            if (g[py]?.[px] !== undefined) g[py][px] = marks[i];
        }
    }
    g[pts[0][1] | 0][pts[0][0] | 0] = '●';
    return g.map(r => r.join('')).join('\n');
}

// ---------- 執行 ----------
const dt = 0.01, rest = fk().map(p => [...p]);
const maxDisp = [0, 0, 0];
for (let step = 0; step < 400; step++) {
    history.push(stepCPG(dt));
    for (const j of joints) stepJoint(j, history[Math.max(0, history.length - 1 - j.delay)] ?? 0, dt);
    const pts = fk();
    pts.slice(1).forEach((p, i) => { maxDisp[i] = Math.max(maxDisp[i], Math.hypot(p[0] - rest[i + 1][0], p[1] - rest[i + 1][1])); });
    if (step === 399 || step === 330) {
        console.log(`第 ${step} 步（█ 上臂、▓ 前臂、▒ 手，● 肩關節）：\n` + render(pts) + '\n');
    }
}
console.log('\n各節末端的最大位移（像素）——越末端動越多：');
joints.forEach((j, i) => console.log(`  ${j.name}末端：${maxDisp[i].toFixed(2)}`));
console.log('\n關節角度是否都在限位內：',
    joints.map(j => `${j.name} ${j.th.toFixed(2)} ∈ [${j.lo}, ${j.hi}]`).join('；'));
