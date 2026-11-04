// handwired 基本做法的最小示範：只用「乘、加、ReLU」，把 if-else 與規則判斷寫成權重。
// 執行：node examples/handwired-basics.mjs
// 不需要任何套件；所有權重都是手寫的，沒有訓練。

const relu = x => Math.max(0, x);

// ---------- 1. 用 ReLU 做出「階梯函數」 ----------
// 整數 a：step(a > 4) = relu(a - 4) - relu(a - 5)，a ≥ 5 時是 1，a ≤ 4 時是 0。
const greaterThan4 = a => relu(a - 4) - relu(a - 5);

// ---------- 2. if-else：兩個分支都算，再用 0/1 選出結果 ----------
// 原本的程式：  if (a > 4) return a * 2; else return a + 10;
const branchy = a => (a > 4 ? a * 2 : a + 10);
// 改寫：兩個分支都算，cond 是 0 或 1，結果 = cond × 分支一 + (1 − cond) × 分支二
// （這裡為了好讀直接相乘；handwired 會再把「乘以 0/1 的選擇」也拆成 ReLU，整個程式只剩矩陣乘法 + ReLU。）
const branchless = a => {
    const cond = greaterThan4(a);
    return (cond * (a * 2)) + ((1 - cond) * (a + 10));
};

// ---------- 3. 夾在 0～1：clamp01(x) = relu(x) − relu(x − 1) ----------
const clamp01 = x => relu(x) - relu(x - 1);

// ---------- 4. 規則判斷寫成一層網路：score = relu(W·f + b) ----------
// 這是 osep-judge 行為分析用的形式（簡化版）。每個樣態是 W 的一列：正權重＝支持，負權重＝反對。
const FEATURES = ['askFirst', 'followSteps', 'adviceMatch', 'aiRequests', 'churnRatio'];
const W = {
    AI_FOLLOWING: [0.5, 1.0, 0.6, 0.6, -0.8], // 照著 AI 做
    TRIAL_AND_ERROR: [0, 0, 0, 0, 1.6] // 反覆試錯（簡化）
};
const B = {AI_FOLLOWING: -0.9, TRIAL_AND_ERROR: -0.5};
const score = (pattern, f) => relu(W[pattern].reduce((sum, w, i) => sum + (w * f[i]), B[pattern]));
const explain = (pattern, f) => FEATURES.map((name, i) => [name, +(W[pattern][i] * f[i]).toFixed(2)])
    .filter(([, value]) => value !== 0);

// ---------- 5. 可微分：權重可以用梯度下降微調 ----------
// 對單一樣態，score = relu(z)，z = W·f + b。z > 0 時 ∂score/∂W_i = f_i、∂score/∂b = 1；z ≤ 0 時梯度是 0。
// 用老師的標註 y（1＝同意、0＝誤判）做一步平方誤差的梯度下降：
const trainStep = (pattern, f, y, lr = 0.1) => {
    const z = W[pattern].reduce((sum, w, i) => sum + (w * f[i]), B[pattern]);
    const s = relu(z);
    const grad = z > 0 ? 2 * (s - y) : 0; // d(s − y)² / dz
    W[pattern] = W[pattern].map((w, i) => w - (lr * grad * f[i]));
    B[pattern] -= lr * grad;
    return {before: s, after: score(pattern, f)};
};

// ---------- 執行 ----------
let ok = true;
for (let a = -3; a <= 12; a++) {
    if (branchy(a) !== branchless(a)) ok = false;
}
console.log(`1-2. if-else 改寫成 ReLU 選擇，a = -3…12 結果與原程式${ok ? '完全相同 ✔' : '不同 ✘'}`);
console.log(`3.   clamp01: ${[-0.5, 0.3, 1.7].map(x => `${x}→${clamp01(x)}`).join('，')}`);

const follower = [0.9, 1.0, 1.0, 1.0, 0.0]; // 沒先試就問、照做、放的都是 AI 提到的、問了很多次、沒拉開刪除
const tinkerer = [0.0, 0.0, 0.1, 0.2, 0.6]; // 自己先試、常常拉開重來
console.log(`4.   照著 AI 做：照做型 ${score('AI_FOLLOWING', follower).toFixed(2)}，自己試型 ${score('AI_FOLLOWING', tinkerer).toFixed(2)}`);
console.log('     照做型的依據（特徵 × 權重）：', explain('AI_FOLLOWING', follower));
const step = trainStep('AI_FOLLOWING', follower, 1);
console.log(`5.   老師標「同意」後做一步梯度下降：分數 ${step.before.toFixed(2)} → ${step.after.toFixed(2)}（往目標 1 靠近）`);
process.exitCode = ok ? 0 : 1;
