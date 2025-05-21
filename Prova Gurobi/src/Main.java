import com.gurobi.gurobi.*;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static final String FILE = "src/Singolo_1.txt";

    public static void main(String[] args) throws IOException, GRBException {

        //Leggo tutti i dati dal file di testo
        double R = readSingleValue("R");
        double E = readSingleValue("E");
        double Q = readSingleValue("Q");
        double z_max = readSingleValue("z_max");
        double[][] teta = readMultipleValue("matrice teta_e^r", (int) R, (int) E);
        double[][] pr = readMultipleValue("pr", 1 ,(int) R);
        double[][] ur = readMultipleValue("ur", 1, (int) R);
        double[][] beta_min = readMultipleValue("beta_min", 1, (int) E);
        double[][] beta_max = readMultipleValue("beta_max", 1, (int) E);
        double rottame_b1 = readSingleValue("r");
        double elemento_b2 = readSingleValue("e");

        GRBEnv env = new GRBEnv();

        GRBModel steel_model = solvePrincipalModel(env, (int) R, (int) E, pr, ur, teta, beta_min, beta_max, z_max, Q);


        System.out.println("\nGRUPPO: Singolo_1");
        System.out.println("Componente: Paolo Formentelli" + "\n");

        //PUNTO [a]
        System.out.println("QUESITO 1:");
        System.out.print("funzione obiettivo: ");
        System.out.printf("%.4f\n", steel_model.get(GRB.DoubleAttr.ObjVal));

        System.out.print("variabili di base: [");
        for (GRBVar base_var: steel_model.getVars()){
            if (base_var.get(GRB.IntAttr.VBasis) == 0)
                System.out.print("1 ");
            else System.out.print("0 ");
        }
        System.out.println("\b]");

        System.out.print("coefficienti di costo ridotto: [");
        for (GRBVar rc: steel_model.getVars()){
            System.out.printf("%.4f ", rc.get(GRB.DoubleAttr.RC));
        }
        System.out.println("\b]");

        System.out.print("Degenere: ");
        boolean degenere = degenere(steel_model);
        if (degenere) System.out.println("SI");
        else System.out.println("NO");

        System.out.print("Soluzione multipla: ");
        boolean multipla = multipla(steel_model);
        if (multipla) System.out.println("SI");
        else System.out.println("NO");

        // Vincoli attivi
        System.out.print("Vincoli attivi: ");
        for (GRBConstr constr : steel_model.getConstrs()) {
            if (Math.abs(constr.get(GRB.DoubleAttr.Slack)) == 0) {
                System.out.print(constr.get(GRB.StringAttr.ConstrName) + " ");
            }
        }
        System.out.println();

        int count = 0;
        for (GRBConstr constr : steel_model.getConstrs()) {
            double slack = steel_model.getConstrByName(constr.get(GRB.StringAttr.ConstrName)).get(GRB.DoubleAttr.Slack);
            if (Math.abs(slack) > 0) {
                count++;
            }
        }
        System.out.println("componenti duale: " + count);
        System.out.println("\n");

        //PUNTO [b]

        System.out.println("QUESITO 2: ");
        printDeltaPr(steel_model, (int) rottame_b1);
        System.out.println();
        printBeta(steel_model, (int)elemento_b2);
        System.out.println();
        System.out.printf("Q = %.4f\n", (Q * (z_max / steel_model.get(GRB.DoubleAttr.ObjVal))));


        //PUNTO [c]
        System.out.println("QUESITO 3:");
        doppiaFase(env, (int) R, (int) E, ur, Q, teta, beta_min, beta_max);


    }

    public static Double readSingleValue(String str) throws IOException {

        File file = new File(FILE);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        String value = "";

        while ((line = br.readLine()) != null){

            line = line.trim();
            if (line.isEmpty()) continue;


            if (line.startsWith(str)) {
                value = line.split("\\s+")[1];
            }
        }

        return Double.parseDouble(value);
    }

    public static double[][] readMultipleValue(String str, int row, int column) throws IOException {

        File file = new File(FILE);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        double[][] value = null;
        boolean start = false;
        int currentLine = 0;


        while ((line = br.readLine()) != null){

            line = line.trim();
            if (line.isEmpty()) continue;


            if (!start && line.startsWith(str)) {
                value = new double[row][column];
                start = true;
                continue;
            }

            if (start && currentLine < row) {
                String[] valori = line.split("\\s+");
                for (int j = 0; j < column; j++) {
                    value[currentLine][j] = Double.parseDouble(valori[j]);
                }
                currentLine++;
            }
        }

        return value;
    }

    //Metodo che ottimizza il problema per il punto [a]
    public static GRBModel solvePrincipalModel(GRBEnv env, int R, int E, double[][] pr, double[][] ur, double[][] teta, double[][] beta_min, double[][] beta_max, Double z_max, Double Q) throws GRBException {
        //Creo ambiente e modello Gurobi
        GRBModel model = new GRBModel(env);
        GRBLinExpr lhs;
        GRBLinExpr rhs;

        //FUNZIONE OBIETTIVO
        GRBLinExpr fo = new GRBLinExpr();
        GRBVar[] r = new GRBVar[R]; //matrice rottami(variabili r)
        for(int i=0; i<R; i++){
            r[i] = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "r_" + (i+1));
            fo.addTerm(pr[0][i], r[i]);
        }
        model.setObjective(fo, GRB.MINIMIZE); // min (p_r1*r1)+...+(p_rR*rR)



        //PRIMI VINCOLI: COMPOSIZIONE CHIMICA ROTTAME
        for (int i=0; i<E; i++){
            lhs = new GRBLinExpr();
            for(int j=0; j<R; j++){
                lhs.addTerm(teta[j][i]/100*ur[0][j]/100, r[j]);
            }
            rhs = new GRBLinExpr();
            rhs.addConstant(beta_min[0][i]/100*Q);
            model.addConstr(lhs, GRB.GREATER_EQUAL, rhs, "vincolo_beta_min_" + (i+1));

            rhs = new GRBLinExpr();
            rhs.addConstant(beta_max[0][i]/100*Q);
            model.addConstr(lhs, GRB.LESS_EQUAL, rhs, "vincolo_beta_max_" + (i+1));
        }


        //SECONDO VINCOLO: PRODUZIONE IN Kg
        rhs = new GRBLinExpr();
        lhs = new GRBLinExpr();
        rhs.addConstant(Q);
        for(int i=0; i<R; i++){
            lhs.addTerm(ur[0][i]/100, r[i]);
        }
        model.addConstr(lhs, GRB.EQUAL, rhs, "vincolo_produzione_Kg");

        model.update();
        model.optimize();

        return model;
    }

    public static boolean degenere(GRBModel model) throws GRBException {
        for (GRBVar var : model.getVars()) {
            if (var.get(GRB.IntAttr.VBasis) == 0 && var.get(GRB.DoubleAttr.X) == 0) {
                return true;
            }
        }
        return false;
    }

    public static boolean multipla(GRBModel model) throws GRBException {
        for (GRBVar var : model.getVars()) {
            if (var.get(GRB.IntAttr.VBasis) != 0 && var.get(GRB.DoubleAttr.RC) == 0) {
                return true;
            }
        }
        return false;
    }

    public static void printDeltaPr(GRBModel model, int r) throws GRBException {
        for(GRBVar var: model.getVars()){
            String nome = "r_"+r;
            if (var.get(GRB.StringAttr.VarName).equals(nome)){ //stampa solo i coefficienti rp=r
                double min = var.get(GRB.DoubleAttr.SAObjLow); //valore minimo e massimo che la variabile può assumere senza cambiare il vertice ottimo
                double max = var.get(GRB.DoubleAttr.SAObjUp);
                System.out.print("variazione di " + var.get(GRB.StringAttr.VarName) + ": "); //Stampo nome variabile pr e il suo delta p
                System.out.printf("[%.4f, %.4f]\n", min, max);
            }
        }
    }

    public static void printBeta(GRBModel model, int e) throws GRBException {
        for (GRBConstr constr : model.getConstrs()) {
            if (constr.get(GRB.StringAttr.ConstrName).startsWith("vincolo_beta_max_"+e)) { // Solo sui vincoli beta_max di e
                double rhs = constr.get(GRB.DoubleAttr.RHS); //Calcola il valore di beta_max
                double min = constr.get(GRB.DoubleAttr.SARHSLow); //clacola i valori di cui può variare beta_max
                double max = constr.get(GRB.DoubleAttr.SARHSUp);
                System.out.print("variazione di " + constr.get(GRB.StringAttr.ConstrName) + ": ");
                System.out.printf("[%.4f  %.4f]\n", (rhs - min), (max - rhs));
            }
        }
    }

    public static void doppiaFase(GRBEnv env, int R, int E, double[][] ur, double Q, double[][] teta, double[][] beta_min, double[][] beta_max) throws GRBException {

        // Creo un nuovo modello Fase 1
        GRBModel modelFase1 = new GRBModel(env);
        modelFase1.set(GRB.StringAttr.ModelName, "fase1");

        GRBVar[] r = new GRBVar[R];
        for (int i = 0; i < R; i++) {
            r[i] = modelFase1.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "x_" + (i+1));
        }

        // Lista per le variabili artificiali
        List<GRBVar> artificial_vars = new ArrayList<>();

        GRBLinExpr lhs = new GRBLinExpr();
        GRBLinExpr rhs = new GRBLinExpr();
        for (int i = 0; i < R; i++) {
            lhs.addTerm(ur[0][i]/100, r[i]);
        }
        GRBVar artificial_max = modelFase1.addVar(0.0, GRB.INFINITY, 1.0, GRB.CONTINUOUS, "art_max");
        lhs.addTerm(1.0, artificial_max);
        rhs.addConstant(Q);
        modelFase1.addConstr(lhs, GRB.EQUAL, rhs, "max_balance");
        artificial_vars.add(artificial_max);


        for (int i = 0; i < E; i++) {
            lhs = new GRBLinExpr();
            rhs = new GRBLinExpr();
            for (int j = 0; j < R; j++) {
                lhs.addTerm(ur[0][j]/100*teta[j][i]/100, r[j]);
            }
            GRBVar artificial_min = modelFase1.addVar(0.0, GRB.INFINITY, 1.0, GRB.CONTINUOUS, "art_beta_min_" + (i+1));
            lhs.addTerm(1.0, artificial_min);
            rhs.addConstant(Q*beta_min[0][i]/100);
            modelFase1.addConstr(lhs, GRB.EQUAL, rhs, "betaMin_" + (i+1));
            artificial_vars.add(artificial_min);

            lhs = new GRBLinExpr();
            rhs = new GRBLinExpr();
            for (int j = 0; j < R; j++) {
                lhs.addTerm(ur[0][j]/100*teta[j][i]/100, r[j]);
            }
            artificial_max = modelFase1.addVar(0.0, GRB.INFINITY, 1.0, GRB.CONTINUOUS, "art_beta_max_" + (i+1));
            lhs.addTerm(1.0, artificial_max);
            rhs.addConstant(Q*beta_max[0][i]/100);
            modelFase1.addConstr(lhs, GRB.EQUAL, rhs, "betaMax_" + (i+1));
            artificial_vars.add(artificial_max);
        }


        modelFase1.optimize();


        System.out.printf("funzione obiettivo = %.4f\n", modelFase1.get(GRB.DoubleAttr.ObjVal));
        System.out.print("valore variabili: [");
        for(int i = 0; i < r.length; i++) {
            System.out.printf("%.4f ", r[i].get(GRB.DoubleAttr.X));
        }
        System.out.println("]");

    }

}