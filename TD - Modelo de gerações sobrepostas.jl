# =============================================================================
# Preâmbulo: Pacotes e Configuração
# =============================================================================
using Random, Distributions, Statistics, StatsBase, Inequality, Plots, Serialization

# =============================================================================
# 1. Definir Parâmetros e Configurações do Modelo
# =============================================================================

const n = 2000                    # número de indivíduos
const T_warm = 30                 # número de períodos de treino
const T_sim = 50                  # número de períodos de simulação (após treino)
const T_total = T_warm + T_sim    # total de períodos a simular

# Parâmetros do modelo (ajuste conforme necessário)
τ   = 0         #Para a versão do modelo com governo, ajustar τ
A   = 1.00      #Para fins de simplificação, essa igualdade será mantida até o final

# Parâmetros para distribuições iniciais
μ₁  = 0.0         # média para δ
μ₂  = 0.0         # média para capital humano inicial h[:,1]

# Momento-alvo para calibração: definir valor alvo (ex: coeficiente de Gini da base de dados)
target_gini = 0.50083 # coeficiente de Gini da renda do trabalho em 2023
gini_tolerance   = 0.01  # diferença aceitável entre Gini simulado e alvo

target_poverty_rate = 0.274 # taxa de pobreza em 2023
poverty_tolerance = 0.01   # diferença aceitável entre taxa de pobreza simulada e alvo

# =============================================================================
# 2. Inicializar Variáveis de Estado
# =============================================================================
# Usamos matrizes com T_total colunas para lidar tanto com o treino quanto com a simulação principal.
h = zeros(n, T_total)   # capital humano dos indivíduos (variável de estado)
c = zeros(n, T_total)   # consumo
e = zeros(n, T_total)   # gasto com educação
c_ot = zeros(n, T_total)   # consumo
e_ot = zeros(n, T_total)   # gasto com educação

# =============================================================================
# 3. Definir a Função de Simulação por Período
# =============================================================================
"""
Simula o período t do modelo e atualiza o estado:
 - Usa a taxa salarial A e calcula um termo de transferência Tr com base no capital humano agregado H.
 - Calcula o consumo e gasto educacional ótimo para cada indivíduo.
 - Atualiza o capital humano para t+1 (se t < T_total).

"""

function simulate_period!(h, c, e, c_ot, e_ot, cmin, t, T_total, δ, β, τ, A, ρ, γ)
    # Capital humano agregado em t
    H_t = sum(h[:, t])
    # Calcula o termo de transferência (Tr) conforme a lógica do modelo
    Tr = (τ / n) * A * H_t

    for i in 1:n
        # Renda do indivíduo i no período t
        income = (1 - τ) * A * h[i, t] + Tr

        # Consumo e educação ótimos conforme equações (5) e (4):
        c_ot[i, t] = income / (1 + β * ρ)
        e_ot[i, t] = β * ρ * income / (1 + β * ρ)

        # Condições de mínimo
        if income < cmin
            c[i, t] = income
            e[i, t] = 0
        elseif c_ot[i, t] < cmin
            c[i, t] = cmin
            e[i, t] = income - cmin
        else
            c[i, t] = c_ot[i, t]
            e[i, t] = e_ot[i, t]   
        end

        # Atualiza capital humano do próximo período conforme equação (8):
        if t < T_total
            if e[i, t] > 0
                h[i, t + 1] = (e[i, t])^ρ * δ[i, t + 1] * γ
            else
                h[i, t + 1] = δ[i, t + 1]
            end
        end
    end
end



# =============================================================================
# 4.1. Grade de Parâmetros: Encontrar combinação que melhor se ajusta ao modelo
# =============================================================================
"""
Nota 1: A primeira versão não separava a seção #4 em duas. Mas por algum motivo, existia uma inconsistência nas variáveis aleatórias.
Mesmo mantendo a seed igual, o comando gerava matrizes diferentes e eu nunca entendi bem o motivo.

A saída para isso foi dividir o script em duas partes: 
    4.1 faz o loop com todas as variáveis; 
    4.2 usa as variâncias encontradas na anterior, define as variáveis aleatórias a priori e refaz o loop;

Com isso, a inconsistência foi resolvida sem prejuízos maiores ao código.


Nota 2: Esta seção demora bastante para rodar, cerca de 8 horas na minha máquina. 
Caso possua interesse de replicar sem passar pela seção, considere inputar os parâmetros manualmente:


β == 0.8
ρ == 0.9
γ == 2.5
c_min == 1.0
σ₁ == 0.3111111111111111
σ₂ == 0.7333333333333333
"""

# Definir grades de busca
betas = range(0.2, 0.8, length=10)
rhos = range(0.1, 0.9, length=10)
gammas = range(1.5, 2.5, length=10)
sigmas1 = range(0.1, 2.0, length=10)
sigmas2 = range(0.1, 2.0, length=10)
cmin_grid = range(1.0, 8.0, length=20) 

# Variáveis auxiliares para guardar os melhores resultados
best_distance_gini = Inf
best_distance_poverty = Inf
best_total_distance = Inf
best_beta = nothing
best_rho = nothing
best_gamma = nothing
best_sigma1 = nothing
best_sigma2 = nothing

best_params = nothing
best_cmin_trial = nothing
best_final_gini = nothing
best_final_poverty = nothing
h_trial = zeros(n, T_warm)

# Loop de calibração
for cmin_trial in cmin_grid
    for β_trial in betas
        for ρ_trial in rhos
            for γ_trial in gammas
                for σ₁_trial in sigmas1
                    for σ₂_trial in sigmas2

                        Random.seed!(1995)
                        h_trial[:,1] = rand(LogNormal(μ₂, σ₂_trial), n)

                        Random.seed!(1995)
                        δ₁_trial = rand(LogNormal(μ₁, σ₁_trial), n, T_warm)

                        c = zeros(n, T_warm)
                        e = zeros(n, T_warm)
                        c_ot = zeros(n, T_warm)
                        e_ot = zeros(n, T_warm)

                        for t in 1:T_warm
                            simulate_period!(h_trial, c, e, c_ot, e_ot, cmin_trial, t, T_warm, δ₁_trial,
                                            β_trial, τ, A, ρ_trial, γ_trial)
                        end

                        final_gini = Inequality.gini(h_trial[:, T_warm])
                        income = (1 - τ) * A .* h_trial[:, T_warm]
                        final_poverty_rate = count(income .< cmin_trial) / n

                        gini_distance = abs(final_gini - target_gini)
                        poverty_distance = abs(final_poverty_rate - target_poverty_rate)

                        norm_gini_dist    = gini_distance / gini_tolerance
                        norm_poverty_dist = poverty_distance / poverty_tolerance

                        total_distance = norm_gini_dist + norm_poverty_dist

                        if gini_distance < gini_tolerance && poverty_distance < poverty_tolerance
                            if total_distance < best_total_distance
                                best_distance_gini = gini_distance
                                best_distance_poverty = poverty_distance
                                best_params = (β_trial, ρ_trial, γ_trial, σ₁_trial, σ₂_trial)
                                best_cmin_trial = cmin_trial
                                best_final_gini = final_gini
                                best_final_poverty = final_poverty_rate
                            end
                        end
                    end
                end
            end
        end
    end
end

if best_params !== nothing
    best_beta, best_rho, best_gamma, best_sigma1, best_sigma2 = best_params
    best_cmin = best_cmin_trial

    println("Melhor β: ", best_beta)
    println("Melhor ρ: ", best_rho)
    println("Melhor γ: ", best_gamma)
    println("Melhor σ₁: ", best_sigma1)
    println("Melhor σ₂: ", best_sigma2)
    println("Melhor cmin: ", best_cmin)
    println("Gini obtido: ", best_final_gini)
    println("Taxa de pobreza final = ", round(best_final_poverty * 100, digits=2), "%")

    # Salvar em arquivo
    open("C:/caminho/best_params_2moments.jld", "w") do io
        serialize(io, (
            beta = best_beta,
            rho = best_rho,
            gamma = best_gamma,           
            cmin = best_cmin,
            final_gini = best_final_gini,
            sigma1 = best_sigma1,
            sigma2 = best_sigma2,
            poverty = best_final_poverty
        ))
    end
else
    println("Nenhum parâmetro adequado encontrado. Tente ajustar a grade ou relaxar a tolerância.")
end

params = deserialize("C:/caminho/best_params_2moments.jld")

best_beta = params.beta
best_rho = params.rho
best_gamma = params.gamma
best_cmin = params.cmin
best_sigma1 = params.sigma1
best_sigma2 = params.sigma2

# =============================================================================
# 4.2. Evitar inconsistências com variáveis aleatórias
# =============================================================================

σ₁ = best_sigma1
σ₂ = best_sigma2

# Gerar os espaços para δ e h
δ = zeros(n, T_total)
h = zeros(n, T_total)

# Preencher δ e h com números aleatórios conforme variância encontrada em 4.1
Random.seed!(1995)
δ = rand(LogNormal(μ₁, σ₁), n, T_total)

Random.seed!(1995)
h[:, 1] = rand(LogNormal(μ₂, σ₂), n) 

# Repetir simulação com δ e h fixos
for cmin_trial in cmin_grid
    for β_trial in betas
        for ρ_trial in rhos
            for γ_trial in gammas       

                c = zeros(n, T_warm)
                e = zeros(n, T_warm)
                c_ot = zeros(n, T_warm)
                e_ot = zeros(n, T_warm)

                for t in 1:T_warm
                    simulate_period!(h, c, e, c_ot, e_ot, cmin_trial, t, T_warm, δ,
                                    β_trial, τ, A, ρ_trial, γ_trial)
                end

                final_gini = Inequality.gini(h_trial[:, T_warm])
                income = (1 - τ) * A .* h_trial[:, T_warm]
                final_poverty_rate = count(income .< cmin_trial) / n

                gini_distance = abs(final_gini - target_gini)
                poverty_distance = abs(final_poverty_rate - target_poverty_rate)

                norm_gini_dist    = gini_distance / gini_tolerance
                norm_poverty_dist = poverty_distance / poverty_tolerance

                total_distance = norm_gini_dist + norm_poverty_dist
                    
                if gini_distance < gini_tolerance && poverty_distance < poverty_tolerance
                    if total_distance < best_total_distance
                        best_distance_gini = gini_distance
                        best_distance_poverty = poverty_distance
                        best_params = (β_trial, ρ_trial, γ_trial)
                        best_cmin_trial = cmin_trial
                        best_final_gini = final_gini
                        best_final_poverty = final_poverty_rate
                    end
                end
            end
        end
    end
end

if best_params !== nothing
    best_beta, best_rho, best_gamma = best_params
    best_cmin = best_cmin_trial

    println("Melhor β: ", best_beta)
    println("Melhor ρ: ", best_rho)
    println("Melhor γ: ", best_gamma)
    println("Melhor σ₁: ", best_sigma1)
    println("Melhor σ₂: ", best_sigma2)
    println("Melhor cmin: ", best_cmin)
    println("Gini obtido: ", best_final_gini)
    println("Taxa de pobreza final = ", round(best_final_poverty * 100, digits=2), "%")

    open("C:/caminho/best_params_teste__2moments_final.jld", "w") do io
        serialize(io, (
            beta = best_beta,
            rho = best_rho,
            gamma = best_gamma,           
            cmin = best_cmin,
            final_gini = best_final_gini,
            sigma1 = best_sigma1,
            sigma2 = best_sigma2,
            poverty = best_final_poverty
        ))
    end
else
    println("Nenhum parâmetro adequado encontrado. Tente ajustar os parâmetros ou aumentar T_warm.")
end

# =============================================================================
# 5. Fase de Treino: Rodar até encontrar os momentos-alvo
# =============================================================================

params = deserialize("C:/Users/joaov/Documents/UFPR/Doutorado/Tese/Computação/Julia/resultados/best_params_teste__2moments_final.jld")

β = params.beta
ρ = params.rho
γ = params.gamma
c_min = params.cmin
σ₁ = params.sigma1
σ₂ = params.sigma2

println("Rodando treino com:")
println("β = ", β)
println("ρ = ", ρ)
println("γ = ", γ)
println("c_min = ", c_min)
println("σ₁ = ", σ₁)
println("σ₂ = ", σ₂)

c = zeros(n, T_total)
e = zeros(n, T_total)
c_ot = zeros(n, T_total)
e_ot = zeros(n, T_total)

h = zeros(n, T_total)
Random.seed!(1995)
h[:, 1] = rand(LogNormal(μ₂, σ₂), n)
δ = rand(LogNormal(μ₁, σ₁), n, T_total)

println("========== Fase de Treino ==========")
match_found = false
warmup_period = T_warm

for t in 1:T_warm
    simulate_period!(h, c, e, c_ot, e_ot, c_min, t, T_total, δ, β, τ, A, ρ, γ)

    current_gini = Inequality.gini(h[:, t])
    income = (1 - τ) * A .* h[:, t]
    current_poverty = count(income .< c_min) / n

    println("Período de treino $t: Gini = $(round(current_gini, digits=8)), Pobreza = $(round(current_poverty, digits=8))")

    gini_distance = abs(current_gini - target_gini)
    poverty_distance = abs(current_poverty - target_poverty_rate)

    if gini_distance < gini_tolerance && poverty_distance < poverty_tolerance
        warmup_period = t
        println("Gini e Pobreza atingidos no período de treino $t.")
        match_found = true
        break
    end
end

if !match_found    
    println("Alvos não atingidos no treino. Tente ajustar os parâmetros ou aumentar T_warm.")
end


# =============================================================================
# 6. Simulação Principal: Continuar a partir do Fim do Treino
# =============================================================================
taus = (0.0, 0.01, 0.03, 0.05)
resultados = Dict()

# Novas séries para renda
pre_tax_income_series  = zeros(n, T_sim)
post_tax_income_series = zeros(n, T_sim)

for τ in taus
    println("Rodando simulação para τ = ", τ)

    # --- Replicar a lógica da SEÇÃO 6 (simulação principal) ---

    T_total = warmup_period + T_sim
    h_new = zeros(n, T_total)
    δ_new = zeros(n, T_total)
    c_new = zeros(n, T_total)
    e_new = zeros(n, T_total)
    c_ot_new = zeros(n, T_total)
    e_ot_new = zeros(n, T_total)

    h_new[:, 1:warmup_period] .= h[:, 1:warmup_period]
    c_new[:, 1:warmup_period] .= c[:, 1:warmup_period]
    e_new[:, 1:warmup_period] .= e[:, 1:warmup_period]
    c_ot_new[:, 1:warmup_period] .= c_ot[:, 1:warmup_period]
    e_ot_new[:, 1:warmup_period] .= e_ot[:, 1:warmup_period]

    δ_new[:, 1:warmup_period] .= δ[:, 1:warmup_period]
    Random.seed!(1995)
    δ_new[:, warmup_period+1:end] .= rand(LogNormal(μ₁, σ₁), n, T_total - warmup_period)

    # Redefinir matrizes principais
    h, δ, c, e, c_ot, e_ot = h_new, δ_new, c_new, e_new, c_ot_new, e_ot_new

    # --- Criar séries temporais ---
    gini_series_post    = zeros(T_total)
    poverty_series_post = zeros(T_total)
    H_series       = zeros(T_total)
    p90_series_post     = zeros(T_total)
    p10_series_post     = zeros(T_total)
    top10_share_post    = zeros(T_total)

    gini_series_pre    = zeros(T_total)
    poverty_series_pre = zeros(T_total)
    p90_series_pre     = zeros(T_total)
    p10_series_pre     = zeros(T_total)
    top10_share_pre    = zeros(T_total)



    for t in warmup_period:(T_total - 1)
        simulate_period!(h, c, e, c_ot, e_ot, c_min, t, T_total, δ, β, τ, A, ρ, γ)

        # Renda pré e pós impostos
        y_pre  = A .* h[:, t]
        Tr     = τ * mean(y_pre)  # transferência per capita
        y_post = (1 - τ) .* y_pre .+ Tr

        # Salvar séries de distribuição
        gini_series_post[t] = Inequality.gini(y_post)
        poverty_series_post[t] = count(y_post .< c_min) / n
        H_series[t] = sum(h[:, t])
        p90_series_post[t] = quantile(y_post, 0.9)
        p10_series_post[t] = quantile(y_post, 0.1)
        top10_income_post = sum(sort(y_post, rev=true)[1:div(n,10)])
        top10_share_post[t] = top10_income_post / sum(y_post)


        gini_series_pre[t] = Inequality.gini(y_pre)
        poverty_series_pre[t] = count(y_pre .< c_min) / n
        p90_series_pre[t] = quantile(y_pre, 0.9)
        p10_series_pre[t] = quantile(y_pre, 0.1)
        top10_income_pre = sum(sort(y_pre, rev=true)[1:div(n,10)])
        top10_share_pre[t] = top10_income_pre / sum(y_pre)


        # Salvar matrizes completas se quiser analisar depois
    #    pre_tax_income_series[:, t - warmup_period + 1] = y_pre
    #   post_tax_income_series[:, t - warmup_period + 1] = y_post
    end

    # Guardar resultados para cada τ
    resultados[τ] = Dict(
        "gini_post" => gini_series_post,
        "poverty_post" => poverty_series_post,
        "p90_post" => p90_series_post,
        "p10_post" => p10_series_post,
        "top10_share_post" => top10_share_post,
        "gini_pre" => gini_series_pre,
        "poverty_pre" => poverty_series_pre,
        "p90_pre" => p90_series_pre,
        "p10_pre" => p10_series_pre,
        "top10_share_pre" => top10_share_pre,
        "Ht" => H_series
    )
end

# =============================================================================
# 7. Pós-Simulação: Plotar e Analisar os Resultados
# =============================================================================

output_dir =  "C:/Users/joaov/Documents/UFPR/Doutorado/Tese/Computação/resultados/Figuras/"

plot_range = warmup_period:(T_total - 1)
sim_range = 1:T_sim

#--------------------------------------------------------- GINI ---------------------------------------------------------
#Gráfico do Gini para cada valor de τ
plot()
for τ in taus
    gini = resultados[τ]["gini_post"]
    plot!(1:T_sim, gini[warmup_period:(T_total-1)],
          label = "τ = $(τ)", lw=2)
end
xlabel!("Período da simulação")
ylabel!("Índice de Gini")
savefig(joinpath(output_dir, "comparacao_gini_taus.png"))



# Gini pré e pós impostos : τ = 0.01
gini_pre_001 = resultados[0.01]["gini_pre"]
gini_post_001 = resultados[0.01]["gini_post"]

plot(1:T_sim, gini_pre_001[plot_range], label="Pré-transferência", color=:red, linestyle=:dash)
plot!(1:T_sim, gini_post_001[plot_range], label="Pós-transferência", color=:red)

savefig(joinpath(output_dir, "gini_tau_001.png"))



# Gini pré e pós impostos : τ = 0.03
gini_pre_003 = resultados[0.03]["gini_pre"]
gini_post_003 = resultados[0.03]["gini_post"]

plot(1:T_sim, gini_pre_003[plot_range], label="Pré-transferência", color=:green, linestyle=:dash)
plot!(1:T_sim, gini_post_003[plot_range], label="Pós-transferência", color=:green)

savefig(joinpath(output_dir, "gini_tau_003.png"))

# Gini pré e pós impostos  τ = 0.05
gini_pre_005 = resultados[0.05]["gini_pre"]
gini_post_005 = resultados[0.05]["gini_post"]

plot(1:T_sim, gini_pre_005[plot_range], label="Pré-transferência", color=:purple, linestyle=:dash)
plot!(1:T_sim, gini_post_005[plot_range], label="Pós-transferência", color=:purple)

savefig(joinpath(output_dir, "gini_tau_005.png"))



#--------------------------------------------------------- TAXA DE POBREZA ---------------------------------------------------------
#Gráfico da taxa de pobreza para cada valor de τ
plot()
for τ in taus
    poverty = resultados[τ]["poverty_post"]
    plot!(1:T_sim, poverty[warmup_period:(T_total-1)],
          label = "τ = $(τ)", lw=2)
end
xlabel!("Período da simulação")
ylabel!("Taxa de pobreza")
savefig(joinpath(output_dir, "comparacao_pobreza_taus.png"))



# Pobreza pré e pós impostos : τ = 0.01
poverty_pre_001 = resultados[0.01]["poverty_pre"]
poverty_post_001 = resultados[0.01]["poverty_post"]

plot(1:T_sim, poverty_pre_001[plot_range], label="Pré-transferência", color=:red, linestyle=:dash)
plot!(1:T_sim, poverty_post_001[plot_range], label="Pós-transferência", color=:red)

savefig(joinpath(output_dir, "poverty_tau_001.png"))


# Pobreza pré e pós impostos : τ = 0.03
poverty_pre_003 = resultados[0.03]["poverty_pre"]
poverty_post_003 = resultados[0.03]["poverty_post"]

plot(1:T_sim, poverty_pre_003[plot_range], label="Pré-transferência", color=:green, linestyle=:dash)
plot!(1:T_sim, poverty_post_003[plot_range], label="Pós-transferência", color=:green)

savefig(joinpath(output_dir, "poverty_tau_003.png"))

# Pobreza pré e pós impostos  τ = 0.05
poverty_pre_005 = resultados[0.05]["poverty_pre"]
poverty_post_005 = resultados[0.05]["poverty_post"]

plot(1:T_sim, poverty_pre_005[plot_range], label="Pré-transferência", color=:purple, linestyle=:dash)
plot!(1:T_sim, poverty_post_005[plot_range], label="Pós-transferência", color=:purple)

savefig(joinpath(output_dir, "poverty_tau_005.png"))

#--------------------------------------------------------- P90/P10 ---------------------------------------------------------
p90p10_ratio_pre = p90_pre ./ p10_pre
p90p10_ratio_post = p90_post ./ p10_post

#Gráfico da taxa de pobreza para cada valor de τ
plot()
for τ in taus
    p90p10_ratio_post = resultados[τ]["p90_post"]./ resultados[τ]["p10_post"]
    plot!(1:T_sim, p90p10_ratio_post[warmup_period:(T_total-1)],
          label = "τ = $(τ)", lw=2)
end
xlabel!("Período da simulação")
ylabel!("P90/P10")
savefig(joinpath(output_dir, "comparacao_p90/p10_taus.png"))




