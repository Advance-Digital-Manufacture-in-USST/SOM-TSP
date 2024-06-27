function route = som_tsp(problem, iterations, learning_rate, color_)
% 自组织映射神经网络求解TSP问题
% 输入量：
% problem ----------- 城市坐标，n行3列的数据
% iterations -------- 迭代次数
% learning_rate ----- 学习率
% color_ ------------ 最终路径的颜色

chect_col = size(problem, 2);
if chect_col ~= 3
    error('请检查第一个参数的输入大小，应为n行3列矩阵')
end

if nargin == 2
    learning_rate = 0.8;
end

% cities = normalize(problem);
cities = problem;
n = size(cities, 1) * 8;
network = generate_network(n);
disp(['产生具有', num2str(n),'个神经元的神经网络，开始迭代'])
for i = 1: iterations
    if ~rem(i, 100)
        disp(['迭代次数：', num2str(i), '/', num2str(iterations)])
    end
    
    city_index = randi(size(cities, 1));
    city = cities(city_index, :);
    winner_index = select_closest(network, city);
    gaussian = get_neighborhood(winner_index, floor(n / 10), size(network, 1));
    network = network + gaussian' .* learning_rate .* (city - network);
    learning_rate = learning_rate * 0.9997;
    n = n * 0.9997;
    
    if ~(mod(i, 1000))
        plot_network(cities, network, i)
    end
    
    if n < 1
        disp(['半径已完全衰退，以', num2str(i), '次迭代完成执行'])
        break
    end
    
    if learning_rate < 0.001
        disp(['学习率已完全衰退，以', num2str(i), '次迭代完成执行'])
        break
    end
end

plot_network(cities, network, i)
route = get_route(cities, network);
plot_route(cities, route, color_)

end



function som_net = generate_network(size_)
som_net = rand(size_, 3);
end


function winner_index = select_closest(candidates, origin)
dis = euclidean_distance(candidates, origin);
[~, winner_index] = min(dis);
end


function distance = euclidean_distance(a, b)
A = a - b;
distance = sqrt(sum(A.^2, 2));
end


function gaussian = get_neighborhood(center, radix, domain)
if radix < 1
    radix = 1;
end

deltas = abs(center - (1: domain));
distance = min(deltas, domain - deltas);
gaussian = exp(-(distance .* distance) ./ (2 * radix * radix));
end


function plot_network(cities, neurons, title_)
figure(2)
scatter3(cities(:, 1), cities(:, 2), cities(:, 3), 'r', '.')
hold on
axis equal
scatter3(neurons(:, 1), neurons(:, 2), neurons(:, 3), 'b', 'o')
title(['第', num2str(title_), '次迭代'])
drawnow
end


function route = get_route(cities, network)
for i = 1: size(cities, 1)
    cities(i, 4) = select_closest(network, cities(i, 1: 3));
end

route = cities(:, 4);
end


function plot_route(cities, route, color_)
cities_ = [cities, route];
cities_sort = sortrows(cities_, 4);

figure(3)
scatter3(cities(:, 1), cities(:, 2), cities(:, 3), 20, 'o', 'r')
hold on
axis equal
grid off
plot3(cities_sort(:, 1), cities_sort(:, 2), cities_sort(:, 3), ...
    'Color', color_, 'LineWidth', 2)
end
