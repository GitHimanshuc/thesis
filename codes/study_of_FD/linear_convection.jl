using Plots
# using FFTW

const δt = 0.1
const δx = 0.01

const tx = Int(δt/δx) # Courant factor



const x_max = 100.0
const t_max = 50.0

const Nx = Int(floor(x_max/δx))+1
const Nt = Int(floor(t_max/δt))+1

xx = 0:δx:((Nx-1)*δx)

const ω1 = 1.0
const ω2 = 10.0
const A1 = 1.0
const A2 = 1.0
const v = .10

const txv = v*tx

function solution(t,x)
    A1*sin(ω1*(x-v*t)) + A2*sin(ω2*(x-v*t))
end



y = zeros(Nt,Nx)

y[1,:] .= solution.(0,xx)
y[2,:] .= solution.(δt,xx)

# plot(y[1,:])

for t in 2:Nt-1
    for x in 2:Nx-1

        y[t+1,x] = (txv)*(y[t,x+1] - y[t,x-1]) + y[t-1,x]
    end

    # Boundary conditions
    y[t+1,1] = solution(t*δt,0)
    y[t+1,Nx] = solution(t*δt,(Nx-1)*δx)
end


plot(y[1,1:100])
for t in 10:10:Nt
    fig = plot!(y[t,1:100])
    display(fig)
end



# Followed plot

function follow(t)
    zeros_pos = Int(v*t*(tx))
    pi_pos = Int(v*t*(tx)) + Int(3/δt)

    zeros_pos:1:pi_pos
end


plot(y[10,follow(10)])
for t in 10:10:Nt
    fig = plot!(y[t,follow(t)])
    display(fig)
end

# NO shift was observed
#
#
# function peaks(y)
#     y[findlocalmaxima(abs.(fft(y)))]
#     # y[]
# end
#
# print(peaks(y[3,:]))
#
# asd = zeros(Nt)
#
#
# for i in 1:Nt
#     asd[i] = peaks(y[i,:])[1]
# end
#
# plot(asd)
#
# n = 200
# asd = abs.(fft(y[n,:]))
# indx = findlocalmaxima(asd)
# print(y[n,indx])
