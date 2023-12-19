#### [蒙特卡罗算法(Monte Carlo method， 也称统计模拟方法)](https://en.wikipedia.org/wiki/Monte_Carlo_method)
> 蒙特卡罗方法在金融工程学、宏观经济学、生物医学、计算物理学（如粒子输运计算、量子热力学计算、空气动力学计算）、机器学习等领域应用广泛。
##### 一、 基本概念
&emsp;&emsp;是使用**随机数**（或者更常见的**伪随机数**）来解决很多计算问题的方法。它的工作原理就是两件事：不断抽样、逐渐逼近。

可粗略地分为两类：
* 一类是所求解的**问题本身具有内在的随机性**，借助计算机的运算能力可以直接模拟这种随机的过程。例如在核物理研究中，分析中子在反应堆中的传输过程。中子与原子核作用受到量子力学规律的制约，人们只能知道它们相互作用发生的概率，却无法准确获得中子与原子核作用时的位置以及裂变产生的新中子的行进速率和方向。科学家依据其概率进行随机抽样得到裂变位置、速度和方向，这样模拟大量中子的行为后，经过*统计*就能获得中子传输的范围，作为反应堆设计的依据。

* 另一种类型是所求解**问题可以转化为某种随机分布的特征数**，比如*随机事件出现的概率*，或者*随机变量的期望值*。通过**随机抽样**的方法，以随机事件出现的频率估计其概率，或者以抽样的数字特征估算随机变量的数字特征，并将其作为问题的解。这种方法多用于**求解复杂的多维积分问题**。

##### 二、 工作过程
在解决实际问题的时候应用蒙特卡罗方法主要有两部分工作：

1. 用蒙特卡罗方法模拟某一过程时，需要产生各种概率分布的随机变量。
2. 用统计方法把模型的数字特征估计出来，从而得到实际问题的数值解。

##### 三、 实际应用
&emsp;&emsp;蒙特卡罗方法尤其适用于模拟输入具有重大不确定性的现象和具有许多耦合自由度的系统。应用领域包括：
*&emsp;&emsp;Monte Carlo methods are especially useful for simulating phenomena with significant uncertainty in inputs and systems with many coupled degrees of freedom. Areas of application include:*

###### 1. 物理学 *Physical sciences*
> *See Also: [Monte Carlo method in statistical physics](https://en.wikipedia.org/wiki/Monte_Carlo_method_in_statistical_mechanics)*

&emsp;&emsp;蒙特卡罗方法在计算物理、物理化学和相关应用领域非常重要，其应用多种多样，从复杂的量子色动力学计算到设计隔热罩和空气动力学形式，以及为辐射剂量学计算建立辐射传输模型。 在辐射材料科学中，模拟离子注入的二元碰撞近似通常基于蒙特卡洛方法来选择下一个碰撞原子。在天体物理学中，蒙特卡洛方法被广泛用于**模拟星系演化**和**微波辐射穿过粗糙行星表面**的情况 。蒙特卡罗方法还被用于构成**现代天气预报基础**的集合模型。
*&emsp;&emsp;Monte Carlo methods are very important in computational physics, physical chemistry, and related applied fields, and have diverse applications from complicated quantum chromodynamics calculations to designing heat shields and aerodynamic forms as well as in modeling radiation transport for radiation dosimetry calculations.*
*&emsp;&emsp;In statistical physics, Monte Carlo molecular modeling is an alternative to computational molecular dynamics, and Monte Carlo methods are used to compute statistical field theories of simple particle and polymer systems. Quantum Monte Carlo methods solve the many-body problem for quantum systems.*
*&emsp;&emsp;In radiation materials science, the binary collision approximation for simulating ion implantation is usually based on a Monte Carlo approach to select the next colliding atom.*
*&emsp;&emsp;In experimental particle physics, Monte Carlo methods are used for designing detectors, understanding their behavior and comparing experimental data to theory.*
*&emsp;&emsp;In astrophysics, they are used in such diverse manners as to model both galaxy evolution and microwave radiation transmission through a rough planetary surface. Monte Carlo methods are also used in the ensemble models that form the basis of modern weather forecasting.*

###### 2. 工程学 *Engineering*
蒙特卡罗方法广泛应用于工程设计中的敏感性分析和定量概率分析。这种需要源于典型过程模拟的交互、共线性和非线性行为。例如:
* 在微电子工程中，蒙特卡罗方法被用于**分析模拟和数字集成电路中的相关和非相关变化**。
* 在地质统计学和地质冶金学中，蒙特卡洛方法是**矿物加工流程设计**的基础，并有助于定量风险分析。
* 在流体动力学领域，特别是**稀薄气体动力学**领域，使用直接模拟蒙特卡罗方法并结合高效计算算法，可求解有限克努森数流体流动的波尔兹曼方程。
* 在自主机器人学中，蒙特卡洛定位可以**确定机器人的位置**。它通常应用于**随机滤波器**，如卡尔曼滤波器或粒子滤波器，后者构成了 SLAM（同步定位和绘图）算法的核心。
* 在电信领域，**规划无线网络**时，必须证明设计适用于多种情况，这些情况主要取决于用户数量、用户位置以及用户希望使用的服务。蒙特卡罗方法通常用于生成这些用户及其状态。然后对网络性能进行评估，如果结果不令人满意，则对网络设计进行优化。

*Monte Carlo methods are widely used in engineering for sensitivity analysis and quantitative probabilistic analysis in process design. The need arises from the interactive, co-linear and non-linear behavior of typical process simulations. For example,*
* *In microelectronics engineering, Monte Carlo methods are applied to analyze correlated and uncorrelated variations in analog and digital integrated circuits.*
* *In geostatistics and geometallurgy, Monte Carlo methods underpin the design of mineral processing flowsheets and contribute to quantitative risk analysis.*
* *In fluid dynamics, in particular rarefied gas dynamics, where the Boltzmann equation is solved for finite Knudsen number fluid flows using the direct simulation Monte Carlo method in combination with highly efficient computational algorithms.*
* *In autonomous robotics, Monte Carlo localization can determine the position of a robot. It is often applied to stochastic filters such as the Kalman filter or particle filter that forms the heart of the SLAM (simultaneous localization and mapping) algorithm.*
* *In telecommunications, when planning a wireless network, the design must be proven to work for a wide variety of scenarios that depend mainly on the number of users, their locations and the services they want to use. Monte Carlo methods are typically used to generate these users and their states. The network performance is then evaluated and, if results are not satisfactory, the network design goes through an optimization process.*
* *In reliability engineering, Monte Carlo simulation is used to compute system-level response given the component-level response.*
* *In signal processing and Bayesian inference, particle filters and sequential Monte Carlo techniques are a class of mean-field particle methods for sampling and computing the posterior distribution of a signal process given some noisy and partial observations using interacting empirical measures.*

###### 3. 气候变化与辐射强迫 *Climate change and radiative forcing*
&emsp;&emsp;政府间气候变化专门委员会依靠蒙特卡罗方法对辐射强迫进行概率密度函数分析。
*&emsp;&emsp;The Intergovernmental Panel on Climate Change relies on Monte Carlo methods in probability density function analysis of radiative forcing.*

###### 4. 计算生物学 *Computational biology*
&emsp;&emsp;蒙特卡罗方法用于计算生物学的各个领域，例如**系统发育中的贝叶斯推断**，或用于研究**基因组、蛋白质或膜等生物系统**。通过计算机模拟，我们可以监控特定分子的局部环境，例如查看是否正在发生某些化学反应。在无法进行物理实验的情况下，可以进行思想实验（例如：断开键、在特定位置引入杂质、改变局部/整体结构或引入外部场）。
*&emsp;&emsp;Monte Carlo methods are used in various fields of computational biology, for example for Bayesian inference in phylogeny, or for studying biological systems such as genomes, proteins, or membranes. The systems can be studied in the coarse-grained or ab initio frameworks depending on the desired accuracy. Computer simulations allow us to monitor the local environment of a particular molecule to see if some chemical reaction is happening for instance.*
*&emsp;&emsp;In cases where it is not feasible to conduct a physical experiment, thought experiments can be conducted (for instance: breaking bonds, introducing impurities at specific sites, changing the local/global structure, or introducing external fields).*

###### 5. 计算机图形学 *Computer graphics*
&emsp;&emsp;路径追踪，有时也被称为**蒙特卡罗光线追踪**，它通过随机追踪可能的光线路径样本来渲染三维场景。对任何给定像素的重复采样最终会使采样的平均值趋近于渲染方程的正确解，从而使其成为目前物理上最精确的三维图形渲染方法之一。
*&emsp;&emsp;Path tracing, occasionally referred to as Monte Carlo ray tracing, renders a 3D scene by randomly tracing samples of possible light paths. Repeated sampling of any given pixel will eventually cause the average of the samples to converge on the correct solution of the rendering equation, making it one of the most physically accurate 3D graphics rendering methods in existence.*

###### 6. 应用统计学 *Applied statistics*
&emsp;&emsp;统计中的蒙特卡罗实验标准是由 Sawilowsky 制定的。在应用统计中，蒙特卡罗方法至少可用于四个目的：

&emsp;&emsp;(1) 在现实数据条件下比较小样本的竞争统计量。虽然在渐近条件下（即样本量无限大和处理效果无限小），可以计算从经典理论分布（如正态曲线、考奇分布）得出的数据的 I 类误差和统计量的幂次属性，但真实数据往往不具备此类分布。
&emsp;&emsp;(2) 提供比精确检验更有效的假设检验实现，如置换检验（通常无法计算），同时比渐近分布的临界值更准确。
&emsp;&emsp;(3) 在贝叶斯推理中提供后验分布的随机样本。该样本近似并概括了后验分布的所有基本特征。
&emsp;&emsp;(4) 为负对数似然函数的 Hessian 矩阵提供有效的随机估计值，这些估计值可以求平均值来形成费雪信息矩阵的估计值。

&emsp;&emsp;蒙特卡罗方法也是近似随机化检验与置换检验之间的折中方法。近似随机化检验基于所有排列的一个指定子集（这可能需要对已考虑的排列进行大量的内务整理）。蒙特卡罗方法则是基于指定数量的排列组合进行测试。
*&emsp;&emsp;The standards for Monte Carlo experiments in statistics were set by Sawilowsky. In applied statistics, Monte Carlo methods may be used for at least four purposes:*

*&emsp;&emsp;1. To compare competing statistics for small samples under realistic data conditions. Although type I error and power properties of statistics can be calculated for data drawn from classical theoretical distributions (e.g., normal curve, Cauchy distribution) for asymptotic conditions (i. e, infinite sample size and infinitesimally small treatment effect), real data often do not have such distributions.*
*&emsp;&emsp;2. To provide implementations of hypothesis tests that are more efficient than exact tests such as permutation tests (which are often impossible to compute) while being more accurate than critical values for asymptotic distributions.*
*&emsp;&emsp;3. To provide a random sample from the posterior distribution in Bayesian inference. This sample then approximates and summarizes all the essential features of the posterior.*
*&emsp;&emsp;4. To provide efficient random estimates of the Hessian matrix of the negative log-likelihood function that may be averaged to form an estimate of the Fisher information matrix.*

*&emsp;&emsp;Monte Carlo methods are also a compromise between approximate randomization and permutation tests. An approximate randomization test is based on a specified subset of all permutations (which entails potentially enormous housekeeping of which permutations have been considered). The Monte Carlo approach is based on a specified number of randomly drawn permutations (exchanging a minor loss in precision if a permutation is drawn twice—or more frequently—for the efficiency of not having to track which permutations have already been selected).*

###### 7. 用于游戏的人工智能 *Artificial intelligence for games*
> Main article: [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

蒙特卡罗方法已被发展成一种称为蒙特卡洛树搜索的技术，可用于**搜索对局中的最佳棋步**。可能的棋步被编排在搜索树中，通过多次随机模拟来估计每一步棋的长期潜力。黑盒模拟器代表对手的棋步。

蒙特卡罗树搜索法（MCTS）有四个步骤：

&emsp;&emsp;(1) 从树的根节点开始，选择最优子节点，直到到达叶节点。
&emsp;&emsp;(2) 展开叶节点，选择其中一个子节点。
&emsp;&emsp;(3) 从该节点开始进行模拟游戏。
&emsp;&emsp;(4) 利用模拟游戏的结果更新节点及其祖先。

在多次模拟对局的过程中，最终的结果是代表一步棋的节点值会上升或下降，希望与该节点是否代表一步好棋相对应。
蒙特卡罗树搜索法已成功用于围棋、Tantrix、Battleship、Havannah 和 Arimaa 等游戏。
*&emsp;&emsp;Monte Carlo methods have been developed into a technique called Monte-Carlo tree search that is useful for searching for the best move in a game. Possible moves are organized in a search tree and many random simulations are used to estimate the long-term potential of each move. A black box simulator represents the opponent's moves.*

*The Monte Carlo tree search (MCTS) method has four steps:*

*&emsp;&emsp;1. Starting at root node of the tree, select optimal child nodes until a leaf node is reached.*
*&emsp;&emsp;2. Expand the leaf node and choose one of its children.*
*&emsp;&emsp;3. Play a simulated game starting with that node.*
*&emsp;&emsp;4. Use the results of that simulated game to update the node and its ancestors.*
*The net effect, over the course of many simulated games, is that the value of a node representing a move will go up or down, hopefully corresponding to whether or not that node represents a good move.*

*Monte Carlo Tree Search has been used successfully to play games such as Go, Tantrix, Battleship, Havannah, and Arimaa.*

> See also: [Computer Go](https://en.wikipedia.org/wiki/Computer_Go)

###### 8. 设计与视觉效果 *Design and visuals*
&emsp;&emsp;蒙特卡洛方法在**求解辐射场和能量传输的耦合积分微分方程**时也很有效，因此这些方法已被用于**全局照明计算**，从而生成虚拟三维模型的逼真图像，并应用于视频游戏、建筑、设计、计算机生成电影和电影特效。
*&emsp;&emsp;Monte Carlo methods are also efficient in solving coupled integral differential equations of radiation fields and energy transport, and thus these methods have been used in global illumination computations that produce photo-realistic images of virtual 3D models, with applications in video games, architecture, design, computer generated films, and cinematic special effects.*

###### 9. 搜寻与救援 *Search and rescue*
&emsp;&emsp;美国海岸警卫队在其计算机建模软件 SAROPS 中使用蒙特卡罗方法**计算搜救行动中船只的可能位置**。每次模拟可生成多达一万个数据点，这些数据点根据所提供的变量随机分布。然后根据这些数据的推断生成搜索模式，以优化围堵概率 (POC) 和探测概率 (POD)，两者相加等于总体成功概率 (POS)。最终，这将成为概率分布的实际应用，以提供最迅速、最便捷的救援方法，从而挽救生命和节省资源。
*&emsp;&emsp;The US Coast Guard utilizes Monte Carlo methods within its computer modeling software SAROPS in order to calculate the probable locations of vessels during search and rescue operations. Each simulation can generate as many as ten thousand data points that are randomly distributed based upon provided variables. Search patterns are then generated based upon extrapolations of these data in order to optimize the probability of containment (POC) and the probability of detection (POD), which together will equal an overall probability of success (POS). Ultimately this serves as a practical application of probability distribution in order to provide the swiftest and most expedient method of rescue, saving both lives and resources.*

###### 10. 金融与商务 *Finance and business*

> See also: [Monte Carlo methods in finance](https://en.wikipedia.org/wiki/Monte_Carlo_methods_in_finance), [Quasi-Monte Carlo methods in finance](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_methods_in_finance), [Monte Carlo methods for option pricing](https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing), [Stochastic modelling (insurance)](https://en.wikipedia.org/wiki/Stochastic_modelling_(insurance)), and [Stochastic asset model](https://en.wikipedia.org/wiki/Stochastic_investment_model)

&emsp;&emsp;蒙特卡罗模拟通常用于评估会影响不同决策选项结果的风险和不确定性。通过蒙特卡罗模拟，商业风险分析师可以将销售量、商品和劳动力价格、利率和汇率等变量的不确定性的总体影响，以及取消合同或改变税法等不同风险事件的影响纳入其中。
&emsp;&emsp;金融领域的蒙特卡罗方法通常用于评估业务单位或公司层面的项目投资，或其他金融估值。蒙特卡罗方法还可用于**期权定价**和**违约风险分析**。此外，蒙特卡罗方法还可用于**估算医疗干预的财务影响**。
*&emsp;&emsp;Monte Carlo simulation is commonly used to evaluate the risk and uncertainty that would affect the outcome of different decision options. Monte Carlo simulation allows the business risk analyst to incorporate the total effects of uncertainty in variables like sales volume, commodity and labour prices, interest and exchange rates, as well as the effect of distinct risk events like the cancellation of a contract or the change of a tax law.*
*&emsp;&emsp;Monte Carlo methods in finance are often used to evaluate investments in projects at a business unit or corporate level, or other financial valuations. They can be used to model project schedules, where simulations aggregate estimates for worst-case, best-case, and most likely durations for each task to determine outcomes for the overall project. Monte Carlo methods are also used in option pricing, default risk analysis. Additionally, they can be used to estimate the financial impact of medical interventions.*

###### 11. 法学 *Law*
&emsp;&emsp;为帮助威斯康星州的女性申请人成功申请骚扰和家庭虐待限制令，我们采用蒙特卡罗方法评估了一项拟议计划的潜在价值。该计划旨在为妇女提供更多的宣传机会，帮助她们申请成功，从而降低遭受强奸和人身攻击的风险。然而，有许多变量无法进行完美估算，包括限制令的有效性、申请者在获得和未获得辩护的情况下的成功率等等。该研究对这些变量进行了不同的试验，以便对整个拟议计划的成功水平做出总体估计。
*&emsp;&emsp;A Monte Carlo approach was used for evaluating the potential value of a proposed program to help female petitioners in Wisconsin be successful in their applications for harassment and domestic abuse restraining orders. It was proposed to help women succeed in their petitions by providing them with greater advocacy thereby potentially reducing the risk of rape and physical assault. However, there were many variables in play that could not be estimated perfectly, including the effectiveness of restraining orders, the success rate of petitioners both with and without advocacy, and many others. The study ran trials that varied these variables to come up with an overall estimate of the success level of the proposed program as a whole.*

###### 12. 图书馆学 *Library science*
&emsp;&emsp;蒙特卡洛方法也被用来模拟马来西亚基于图书类型的图书出版数量。蒙特卡洛模拟利用了以前出版的国家图书出版数据和当地市场上不同类型图书的价格。蒙特卡洛模拟的结果被用来确定马来西亚人喜欢哪种类型的图书，并被用来比较马来西亚和日本的图书出版情况。
*&emsp;&emsp;Monte Carlo approach had also been used to simulate the number of book publications based on book genre in Malaysia. The Monte Carlo simulation utilized previous published National Book publication data and book's price according to book genre in the local market. The Monte Carlo results were used to determine what kind of book genre that Malaysians are fond of and was used to compare book publications between Malaysia and Japan.*

###### 13. 其他 *Other*
&emsp;&emsp;纳西姆-尼古拉斯-塔勒布（Nassim Nicholas Taleb）在其 2001 年出版的《被随机性愚弄》（Fooled by Randomness）一书中写道，蒙特卡洛生成器是反向图灵测试的一个真实案例：如果一个人的书写无法与生成的书写区分开来，就可以宣布他是不聪明的。
*&emsp;&emsp;Nassim Nicholas Taleb writes about Monte Carlo generators in his 2001 book Fooled by Randomness as a real instance of the reverse Turing test: a human can be declared unintelligent if their writing cannot be told apart from a generated one.*

##### 四、 在数学中的应用
>一般来说，蒙特卡罗方法用于数学领域，通过生成合适的随机数（另见随机数生成）并观察其中符合某种或某些属性的那部分数来解决各种问题。对于过于复杂而无法分析求解的问题，该方法有助于获得数值解。蒙特卡罗法最常见的应用是蒙特卡罗积分法。

###### 1. 积分 *Integration*
> Main article: [Monte Carlo integration](https://en.wikipedia.org/wiki/Monte_Carlo_integration)

&emsp;&emsp;确定性数值积分算法在维数较少的情况下运行良好，但当函数具有多个变量时，就会遇到两个问题。首先，所需的函数求值次数随着维数的增加而迅速增加。例如，如果在一个维度中 10 次求值就能提供足够的精度，那么在 100 个维度中就需要 10100 个点--计算量太大了。这就是所谓的 "维度诅咒"。其次，多维区域的边界可能非常复杂，因此将问题简化为迭代积分可能并不可行。100 维并不罕见，因为在许多物理问题中，一个 "维 "相当于一个自由度。
*&emsp;&emsp;Deterministic numerical integration algorithms work well in a small number of dimensions, but encounter two problems when the functions have many variables. First, the number of function evaluations needed increases rapidly with the number of dimensions. For example, if 10 evaluations provide adequate accuracy in one dimension, then 10100 points are needed for 100 dimensions—far too many to be computed. This is called the curse of dimensionality. Second, the boundary of a multidimensional region may be very complicated, so it may not be feasible to reduce the problem to an iterated integral. 100 dimensions is by no means unusual, since in many physical problems, a "dimension" is equivalent to a degree of freedom.*

&emsp;&emsp;蒙特卡罗方法提供了一种摆脱计算时间指数增长的方法。只要有关函数的表现相当良好，就可以通过在 100 维空间中随机选取点，并对这些点上的函数值取某种平均值来估算。根据中心极限定理，这种方法会显示 $1/\sqrt(N)$ 收敛性--也就是说，无论维数多少，采样点数量增加四倍，误差就会减半。
*&emsp;&emsp;Monte Carlo methods provide a way out of this exponential increase in computation time. As long as the function in question is reasonably well-behaved, it can be estimated by randomly selecting points in 100-dimensional space, and taking some kind of average of the function values at these points. By the central limit theorem, this method displays $1/\sqrt(N)$ convergence—i.e., quadrupling the number of sampled points halves the error, regardless of the number of dimensions.*

&emsp;&emsp;这种方法的改进版在统计学中被称为重要度抽样，涉及随机抽样点，但在积分较大的情况下更为频繁。要精确地做到这一点，必须已经知道积分，但可以用类似函数的积分来近似积分，或使用自适应例程，如分层抽样、递归分层抽样、自适应伞状抽样或 VEGAS 算法。
*&emsp;&emsp;A refinement of this method, known as importance sampling in statistics, involves sampling the points randomly, but more frequently where the integrand is large. To do this precisely one would have to already know the integral, but one can approximate the integral by an integral of a similar function or use adaptive routines such as stratified sampling, recursive stratified sampling, adaptive umbrella sampling or the VEGAS algorithm.*

&emsp;&emsp;一种类似的方法，即准蒙特卡洛法，使用低差异序列。这些序列能更好地 "填充 "区域，更频繁地对最重要的点进行采样，因此准蒙特卡罗方法通常能更快地收敛于积分。
*&emsp;&emsp;A similar approach, the quasi-Monte Carlo method, uses low-discrepancy sequences. These sequences "fill" the area better and sample the most important points more frequently, so quasi-Monte Carlo methods can often converge on the integral more quickly.*

&emsp;&emsp;另一种对体积中的点进行采样的方法是在体积上模拟随机行走（马尔科夫链蒙特卡罗）。这类方法包括 Metropolis-Hastings 算法、Gibbs 采样、Wang 和 Landau 算法，以及交互式 MCMC 方法，如顺序蒙特卡罗采样器。
*&emsp;&emsp;Another class of methods for sampling points in a volume is to simulate random walks over it (Markov chain Monte Carlo). Such methods include the Metropolis–Hastings algorithm, Gibbs sampling, Wang and Landau algorithm, and interacting type MCMC methodologies such as the sequential Monte Carlo samplers.*

###### 2. 模拟与优化 *Simulation and optimization*
> Main article: [Stochastic optimization](https://en.wikipedia.org/wiki/Stochastic_optimization)

&emsp;&emsp;随机数在数值模拟中的另一个强大且非常流行的应用是数值优化。问题是**如何最小化（或最大化）某些向量的函数**，而这些向量通常有很多维度。许多问题都可以这样表述：例如，计算机国际象棋程序可以被视为试图找到一组，例如 10 步棋，最后产生最佳评价函数的程序。在旅行推销员问题中，目标是使旅行距离最小化。工程设计中也有应用，如**多学科设计优化**。它已被应用于准一维模型，通过有效探索大型配置空间来解决粒子动力学问题。[参考文献](https://www.jhuapl.edu/ISSO/)全面回顾了与仿真和优化相关的许多问题。
*&emsp;&emsp;Another powerful and very popular application for random numbers in numerical simulation is in numerical optimization. The problem is to minimize (or maximize) functions of some vector that often has many dimensions. Many problems can be phrased in this way: for example, a computer chess program could be seen as trying to find the set of, say, 10 moves that produces the best evaluation function at the end. In the traveling salesman problem the goal is to minimize distance traveled. There are also applications to engineering design, such as multidisciplinary design optimization. It has been applied with quasi-one-dimensional models to solve particle dynamics problems by efficiently exploring large configuration space. Reference is a comprehensive review of many issues related to simulation and optimization.*

&emsp;&emsp;**旅行推销员问题**是所谓的传统优化问题。也就是说，确定最佳路径所需的所有事实（每个目的地之间的距离）都是确定无疑的，我们的目标是对可能的旅行选择进行排序，选出总距离最小的那个。但是，假设我们不是要尽量减少到达每个目的地的总路程，而是要尽量减少到达每个目的地所需的总时间。这超出了传统优化的范围，因为旅行时间本身就是不确定的（交通堵塞、时间等）。因此，为了确定我们的最佳路径，我们需要使用模拟-优化方法，首先了解从一点到另一点可能需要的时间范围（在本例中用概率分布而不是具体距离表示），然后优化我们的旅行决策，在考虑到这种不确定性的情况下确定最佳路径。
*&emsp;&emsp;The traveling salesman problem is what is called a conventional optimization problem. That is, all the facts (distances between each destination point) needed to determine the optimal path to follow are known with certainty and the goal is to run through the possible travel choices to come up with the one with the lowest total distance. However, let's assume that instead of wanting to minimize the total distance traveled to visit each desired destination, we wanted to minimize the total time needed to reach each destination. This goes beyond conventional optimization since travel time is inherently uncertain (traffic jams, time of day, etc.). As a result, to determine our optimal path we would want to use simulation – optimization to first understand the range of potential times it could take to go from one point to another (represented by a probability distribution in this case rather than a specific distance) and then optimize our travel decisions to identify the best path to follow taking that uncertainty into account.*

###### 3. 反向问题 *Inverse problems*

&emsp;&emsp;逆问题的概率表述导致模型空间中概率分布的定义。该概率分布结合了先验信息和通过测量某些可观测参数（数据）获得的新信息。在一般情况下，将数据与模型参数联系起来的理论是非线性的，因此模型空间中的**后验概率可能不容易描述**（可能是多模态的，某些矩可能没有定义等）。
*&emsp;&emsp;Probabilistic formulation of inverse problems leads to the definition of a probability distribution in the model space. This probability distribution combines prior information with new information obtained by measuring some observable parameters (data). As, in the general case, the theory linking data with model parameters is nonlinear, the posterior probability in the model space may not be easy to describe (it may be multimodal, some moments may not be defined, etc.).*

&emsp;&emsp;在分析逆问题时，**获得最大似然模型**通常是**不够**的，因为我们通常还希望获得有关数据解析力的信息。在一般情况下，我们可能会有很多模型参数，而对感兴趣的边际概率密度进行检验可能不切实际，甚至毫无用处。但是，我们可以**根据后验概率分布伪随机生成大量模型**，并对模型进行分析和显示，从而将模型属性的相对可能性信息传递给观众。即使在没有明确的先验分布公式的情况下，也可以通过高效的蒙特卡罗方法实现这一目标。
*&emsp;&emsp;When analyzing an inverse problem, obtaining a maximum likelihood model is usually not sufficient, as we normally also wish to have information on the resolution power of the data. In the general case we may have many model parameters, and an inspection of the marginal probability densities of interest may be impractical, or even useless. But it is possible to pseudorandomly generate a large collection of models according to the posterior probability distribution and to analyze and display the models in such a way that information on the relative likelihoods of model properties is conveyed to the spectator. This can be accomplished by means of an efficient Monte Carlo method, even in cases where no explicit formula for the a priori distribution is available.*

&emsp;&emsp;最著名的重要性采样方法--Metropolis 算法--可以通用化，这就提供了一种方法，可以分析具有复杂先验信息和任意噪声分布数据的（可能是高度非线性的）逆问题。
*&emsp;&emsp;The best-known importance sampling method, the Metropolis algorithm, can be generalized, and this gives a method that allows analysis of (possibly highly nonlinear) inverse problems with complex a priori information and data with an arbitrary noise distribution.*

###### 4. 哲学 *Philosophy*

&emsp;&emsp;麦克莱肯对蒙特卡罗方法进行了通俗阐述。埃利萨科夫和格吕内-亚诺夫及魏里希对该方法的一般理念进行了讨论。
*&emsp;&emsp;Popular exposition of the Monte Carlo Method was conducted by McCracken. The method's general philosophy was discussed by Elishakoff and Grüne-Yanoff and Weirich.*

##### 五、 $See \ Also$
* [Auxiliary field Monte Carlo](https://en.wikipedia.org/wiki/Auxiliary-field_Monte_Carlo)
* [Biology Monte Carlo method](https://en.wikipedia.org/wiki/Biology_Monte_Carlo_method)
* [Direct simulation Monte Carlo](https://en.wikipedia.org/wiki/Direct_simulation_Monte_Carlo)
* [Dynamic Monte Carlo method](https://en.wikipedia.org/wiki/Dynamic_Monte_Carlo_method)
* [Ergodicity（遍历理论）](https://en.wikipedia.org/wiki/Ergodic_theory)
* [Genetic algorithms](https://en.wikipedia.org/wiki/Genetic_algorithm)
* [Kinetic Monte Carlo](https://en.wikipedia.org/wiki/Kinetic_Monte_Carlo)
* [List of software for Monte Carlo molecular modeling](https://en.wikipedia.org/wiki/List_of_software_for_Monte_Carlo_molecular_modeling)
* [Mean-field particle methods](https://en.wikipedia.org/wiki/Mean-field_particle_methods)
* [Monte Carlo method for photon transport](https://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport)
* [Monte Carlo methods for electron transport](https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_electron_transport)
* [Monte Carlo N-Particle Transport Code](https://en.wikipedia.org/wiki/Monte_Carlo_N-Particle_Transport_Code)
* [Morris method（莫里斯方法）](https://en.wikipedia.org/wiki/Morris_method)
* [Multilevel Monte Carlo method](https://en.wikipedia.org/wiki/Multilevel_Monte_Carlo_method)
* [Quasi-Monte Carlo method](https://en.wikipedia.org/wiki/Quasi-Monte_Carlo_method)
* [Sobol sequence（Sobol' 序列，伪蒙特卡罗方法）](https://en.wikipedia.org/wiki/Sobol_sequence)
* [Temporal difference learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)