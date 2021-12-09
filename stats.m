load("matdata")

rl_mean = mean(rldata);
rl_std = std(rldata);
rl_var = var(rldata);

nrl_mean = mean(nonrldata);
nrl_std = std(nonrldata);
nrl_var = var(nonrldata);

min_rl = min(rldata)
max_nonRl = max(nonrldata)

count = 0;
count2 = 0;
count3 = 0;
for i = 1:size(rldata)
    if rldata(i) >= rl_mean
        count = count + 1;
    end
    if rldata(i) >= nrl_mean
        count2 = count2 + 1;
    end
    if rldata(i) <= max_nonRl
        count3 = count3 + 1;
    end
end
count/i
count2/i
count3/i