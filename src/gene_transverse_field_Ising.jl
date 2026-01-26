using LinearAlgebra
using DelimitedFiles
using HDF5

saveData(data,filename)= open(filename,"w") do io writedlm(io,data) end
loadData(filename)=readdlm(filename)
linspace(start,stop,length)=range(start,stop=stop,length=length)

function gene_pauli()
    idop=diagm([1.0,1.0])
    σxop=[0 1.0; 1.0 0]
    σyop=[0 -1.0im; 1.0im 0]
    σzop=diagm([1.0,-1.0])
    σs=[idop,σxop,σyop,σzop]
end

σs=gene_pauli()
idop,σxop,σyop,σzop=σs

function promoteOp(op,idx::Integer,num::Integer)
#function promoteOp(op,idx::Tuple,num::Integer) # Edward, edited to fix python error for edges being a tuple
    ops=Vector{Any}([idop for _ in 1:num])
    ops[idx]=op
    kron(ops...)
end

function promoteOp(op,idx::Vector,num::Integer)
    ops=Vector{Any}([idop for _ in 1:num])
    for (pos_,idx_ ) in enumerate(idx)
        ops[idx_]=op[pos_]
    end    
    kron(ops...)
end


function cal_Hzz_Hx_general(Jij,hx,nqubit)
    Hzz=zeros(2^nqubit,2^nqubit)
    for (key,val) in Jij
        #Hzz+=promoteOp([σzop,σzop],key,nqubit)*val
        Hzz+=promoteOp([σzop,σzop],collect(key),nqubit)*val # add collect to convert key to a vector
    end
    Hx=zeros(2^nqubit,2^nqubit)
    for i in 1:nqubit
        Hx+=promoteOp(σxop,i,nqubit)*hx[i]
    end
    Hzz,Hx
end

function compute_ρ(H)
    val,vec=eigen(H)
    gs_vec=vec[:,1]
    ρ=gs_vec*conj(transpose(gs_vec))
end


function cal_obs_x_zz(ρ,nqubit,edges)
    x_obs=[real(tr(ρ*promoteOp([σxop],[i],nqubit)))  for i in 1:nqubit]
    zz_obs=[real(tr(ρ*promoteOp([σzop,σzop],[i,j],nqubit)))  for (i,j) in edges ]
    x_obs,zz_obs
end



# original function

function geneData(θs,nqubit,edges)
    Jij=Dict([edge=>(rand()-0.5) for  edge in edges])
    Jijval=[Jij[edge] for edge in edges]
    Jijvalnorm=sqrt(sum(Jijval.^2))
    Jijvalscaled=Jijval/Jijvalnorm
    hxval=rand(nqubit)

    #println("Types of data within geneData:")
    #@show typeof(Jij)
    #@show typeof(hxval)
    #println("end")
    #exit(0)

    Hzz,Hx=cal_Hzz_Hx_general(Jij,hxval,nqubit)
    inputs=[]
    outputs=[]
    for θ in θs
        H=sin(θ)*Hzz-cos(θ)*Hx
        # H = Hzz - Hx
        ρ=compute_ρ(H)
        x_obs,zz_obs=cal_obs_x_zz(ρ,nqubit,edges)
        input=[x_obs...,Jijvalscaled...]
        output=zz_obs
        push!(inputs,input)
        push!(outputs,output)
    end
    inputs=hcat(inputs...)
    outputs=hcat(outputs...)
    inputs,outputs, Jij, hxval
end 

# needs to be supplied unnormalized inputs.
# J spans -1 to 1, hx: 0 to 1, and theta: 0 to pi/2
function oracle(J,h,θ,nqubit,edges)
    Hzz,Hx=cal_Hzz_Hx_general(J,h,nqubit)
    H=sin(θ)*Hzz-cos(θ)*Hx
    ρ=compute_ρ(H)
    x_obs, zz_obs=cal_obs_x_zz(ρ,nqubit,edges)
    return x_obs, zz_obs
end

#=
function oracle(Jij,hxval,θs,nqubit,edges)
    # modified geneData function by Edward, starts from solving for observables
    # saves Jij, hx to inputs and x_obs and zz_obs to the output
    # removed normalization of Jij in the data, can do this in python after.

    Hzz,Hx=cal_Hzz_Hx_general(Jij,hxval,nqubit)
    
    # save flattened Jij out
    Jijvec = [Jij[e] for e in edges]
    input = vcat(Jijvec,hxval)

    #inputs=[]
    #outputs=[]
    #push!(inputs,input)
    
    #outs = Vector{Vector{Float64}}()
    outs = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, length(θs))

    #for θ in θs
    for (k,θ) in pairs(θs)
        H=sin(θ)*Hzz-cos(θ)*Hx
        ρ=compute_ρ(H)
        x_obs,zz_obs=cal_obs_x_zz(ρ,nqubit,edges)
        #input=[Jij...,hxval...]
        #output=[x_obs,zz_obs] # added x_obs to output
        #output=[x_obs...,zz_obs...] # added x_obs to output
        #push!(outputs,output)
        #push!(outs,vcat(x_obs,zz_obs))
        outs[k] = (x_obs,zz_obs)
    end
    return input,outs
end
=#

# Notes (Edward): want to execute this main part in a python script oracle-style
# commenting out the main/data saving section from Zhengqiang's code, will handle in AL pipeline in python



#=
println("Beginning...")

nqubit=4
Nθ=200
θs=linspace(0,pi/2.0,Nθ)
edges=[ [i,j] for i in 1:nqubit for j in 1:nqubit if i<j]

println("Number of edges:")
println(edges)

println("Original types:")
#@show typeof(Jij)
#@show typeof(hxval)
@show typeof(θs)
@show typeof(nqubit)
@show typeof(edges), eltype(edges)
println("end")

inputs, outputs, Jij, hxval = geneData(θs,nqubit,edges)

println("Types of data within geneData:")
@show typeof(Jij)
@show typeof(hxval)
println("end")

data_dir="./test_$(nqubit)"
mkdir(data_dir)

h5open("$(data_dir)/train_nqubit_$(nqubit).h5","w") do file
    #for idx in 1:100000
    for idx in 1:1000
        print("$(idx)\n")
        inputs,outputs=geneData(θs,nqubit,edges)
        write(file, "input_$(idx)", inputs)
        write(file, "output_$(idx)", outputs)
    end
end


h5open("$(data_dir)/test_nqubit_$(nqubit).h5","w") do file
    #for idx in 1:10000
    for idx in 1:100
        print("$(idx)\n")
        inputs,outputs=geneData(θs,nqubit,edges)
        write(file, "input_$(idx)", inputs)
        write(file, "output_$(idx)", outputs)
    end
end



input_test=h5open("$(data_dir)/test_nqubit_$(nqubit).h5") do file
    read(file,"input_1")
end

output_test=h5open("$(data_dir)/test_nqubit_$(nqubit).h5") do file
    read(file,"output_1")
end

=#

println("Main file loaded..")

